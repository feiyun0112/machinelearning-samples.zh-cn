# 垃圾短信检测

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4           | 动态API | 可能需要更新项目结构以匹配模板 | 控制台应用程序 | .tsv 文件 | 垃圾信息检测 | 二元分类 | Averaged Perceptron（线性学习器）|

在这个示例中，您将看到如何使用[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)来预测短信是否是垃圾信息。在机器学习领域中，这种类型的预测被称为**二元分类**。 

## 问题

我们的目标是预测一个短信是否是垃圾信息（一个不相关的/不想要的消息）。我们将使用UCI的[SMS Spam Collection Data Set](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)，其中包含近6000条被分类为“垃圾信息”或“ham”（不是垃圾信息）的消息。我们将使用这个数据集来训练一个模型，该模型可以接收新消息并预测它们是否是垃圾信息。

这是一个二元分类的示例，因为我们将短信分类为两个类别。

## 解决方案
要解决这个问题，首先我们将建立一个评估器来定义我们想要使用的机器学习管道。 然后，我们将在现有数据上训练这个评估器，评估其有多好，最后我们将使用该模型来预测一些示例消息是否是垃圾信息。

![建立 -> 训练 -> 评估 -> 使用](../shared_content/modelpipeline.png)

### 1. 建立模型

为了建立模型，我们将：

* 定义如何读取从 https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection 下载的垃圾信息数据集。 

* 应用多个数据转换：

    * 将标签（“spam”或“ham”）转换为布尔值（“true”表示垃圾信息），这样我们就可以在二元分类器中使用它。    
    * 将短信转换为数字向量，以便机器学习训练器可以使用它 
    
* 添加一个训练器（如`StochasticDualCoordinateAscent`）。

初始代码类似以下内容：

```CSharp
// Set up the MLContext, which is a catalog of components in ML.NET.
MLContext mlContext = new MLContext();

// Specify the schema for spam data and read it into DataView.
var data = mlContext.Data.LoadFromTextFile<SpamInput>(path: TrainDataPath, hasHeader: true, separatorChar: '\t');

// Data process configuration with pipeline data transformations 
var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                                      .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesText", new Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options
                                      {
                                          WordFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options { NgramLength = 2, UseAllLengths = true },
                                          CharFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options { NgramLength = 3, UseAllLengths = false },
                                      }, "Message"))
                                      .Append(mlContext.Transforms.CopyColumns("Features", "FeaturesText"))
                                      .Append(mlContext.Transforms.NormalizeLpNorm("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);

// Set the training algorithm 
var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", numberOfIterations: 10, featureColumnName: "Features"), labelColumnName: "Label")
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
var trainingPipeLine = dataProcessPipeline.Append(trainer);
```

### 2. 评估模型

对于这个数据集，我们将使用[交叉验证](https://en.wikipedia.org/wiki/Cross-validation_(statistics))来评估我们的模型。将数据集划分成5个不相交的子集，训练5个模型（每个模型使用其中4个子集），并在训练中没有使用的数据子集上测试模型。

```CSharp
var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(data: data, estimator: trainingPipeLine, numberOfFolds: 5);
```

请注意，通常我们在训练后评估模型。 但是，交叉验证包括模型训练部分，因此我们不需要先执行`Fit()`。 但是，我们稍后将在完整数据集上训练模型以利用其他数据。

### 3. 训练模型
为了训练模型，我们将调用评估器的`Fit()`方法，同时提供完整的训练数据。

```CSharp
var model = trainingPipeLine.Fit(data);
```

### 4. 使用模型

训练完模型后，您可以使用`Predict()`API来预测新文本是否垃圾信息。 在这种情况下，我们更改模型的阈值以获得更好的预测。 我们这样做是因为我们的数据有偏差，大多数消息都不是垃圾信息。

```CSharp
//Create a PredictionFunction from our model 
var predictor = mlContext.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(model);

var input = new SpamInput { Message = "free medicine winner! congratulations" };
Console.WriteLine("The message '{0}' is {1}", input.Message, prediction.isSpam == "spam" ? "spam" : "not spam");

```
