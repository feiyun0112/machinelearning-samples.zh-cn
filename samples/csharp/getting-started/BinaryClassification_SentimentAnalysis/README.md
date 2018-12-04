# 用户评论的情绪分析 

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v0.7           | 动态API | README.md 已更新 | 控制台应用程序 | .tsv 文件 | 情绪分析 | 二元分类 | 线性分类 |

在这个介绍性示例中，您将看到如何使用[ML.NET](https://www.microsoft.com/net/./apps/machine-.-and-ai/ml-dotnet)预测客户评论的情绪（积极或消极）。在机器学习领域中，这种类型的预测被称为**二元分类**。

## 问题
这个问题集中在预测客户的评论是否具有正面或负面情绪。我们将使用小型的wikipedia-detox-datasets（一个用于训练的数据集，一个用于模型的准确性评估的数据集），这些数据集已经由人工处理过，并且每个评论都被分配了一个情绪标签：
* 0 - 好评/正面
* 1 - 差评/负面

我们将使用这些数据集构建一个模型，在预测时将分析字符串并预测情绪值为0或1。

## 机器学习任务 - 二元分类
**二元分类**一般用于将项目分类为两个类中的一个的问题（将项目分类为两个以上的类称为**多类分类**）。

* 预测保险索赔是否有效。
* 预测飞机是否会延误或将准时到达。
* 预测face ID（照片）是否属于设备的所有者。 

所有这些示例的共同特征是我们想要预测的参数只能采用两个值中的一个。 换句话说，该值由 `boolean` 类型表示。

## 解决方案
要解决这个问题，首先我们将建立一个机器学习模型。然后，我们将在现有数据上训练模型，评估其有多好，最后我们将使用该模型来预测新评论的情绪。 

![建立 -> 训练 -> 评估 -> 使用](../shared_content/modelpipeline.png)

### 1. 建立模型

建立模型包括：

* 定义映射到数据集的数据架构，以便用DataReader读取(“wikipedia-detox-250-line-data.tsv”和“wikipedia-detox-250-line-test.tsv”)

* 创建一个评估器，并将数据转换为数值向量，以便它能够被机器学习算法有效地使用（使用“FeaturizeText”）

* 选择训练器/学习算法(如“FastTree”)来训练模型。

初始代码类似以下内容：

```CSharp
// STEP 1: Common data loading configuration
TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
                                        {
                                            Separator = "tab",
                                            HasHeader = true,
                                            Column = new[]
                                                        {
                                                        new TextLoader.Column("Label", DataKind.Bool, 0),
                                                        new TextLoader.Column("Text", DataKind.Text, 1)
                                                        }
                                        });
IDataView trainingDataView = textLoader.Read(TrainDataPath);
IDataView testDataView = textLoader.Read(TestDataPath);

// STEP 2: Common data process configuration with pipeline data transformations          
var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Text", "Features");

// STEP 3: Set the training algorithm, then create and config the modelBuilder                            
var trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumn: "Label", featureColumn: "Features");
var trainingPipeline = dataProcessPipeline.Append(trainer);
```

### 2. 训练模型
训练模型是在训练数据（具有已知情绪值）上运行所选算法以调整模型参数的过程。它是在评估器对象的 `Fit()` 方法中实现。

为了执行训练，您需要在DataView对象中提供了训练数据集（`wikipedia-detox-250-line-data.tsv`文件）后调用 `Fit()` 方法。

```CSharp
ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);
```

请注意，ML.NET使用延迟加载方式处理数据，所以在实际调用.Fit()方法之前，没有任何数据真正加载到内存中。

### 3. 评估模型

我们需要这一步骤来判定我们的模型对新数据的准确性。 为此，上一步中的模型再次针对另一个未在训练中使用的数据集（`wikipedia-detox-250-line-test.tsv`）运行。 此数据集也包含了已知的情绪。

`Evaluate()`比较测试数据集的预测值，并生成各种指标，例如准确性，您可以对其进行浏览。 

```CSharp
var predictions = trainedModel.Transform(testDataView);
var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label", "Score");

ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
```

如果您对模型的质量不满意，可以通过提供更大的训练数据集，并为每个算法选择具有不同超参数的不同训练算法来尝试改进它。

>*请记住，对于这个示例，它的质量会低于可能的质量，因为数据集的大小被减少了，以便可以很快地训练。您应该使用更大的已标记情绪的数据集来显著提高模型的质量。*

### 4. 使用模型

训练完模型后，您可以使用`Predict()`API来预测新示例文本的情绪。

```CSharp
// Create prediction engine related to the loaded trained model
var predFunction= trainedModel.MakePredictionFunction<SentimentIssue, SentimentPrediction>(mlContext);

//Score
var resultprediction = predFunction.Predict(sampleStatement);
```

其中`resultprediction.PredictionLabel`将为True或False，具体取决于它是否被预测为负面或正面的情绪。
