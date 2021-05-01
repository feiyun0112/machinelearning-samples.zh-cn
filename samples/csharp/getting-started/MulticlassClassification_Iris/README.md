# 鸢尾花分类

ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4           | 动态 API | 最新版本 | 控制台应用程序 | .txt 文件 | 鸢尾花分类 | 多类分类 | Sdca Multi-class |

在这个介绍性示例中，您将看到如何使用[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)来预测鸢尾花的类型。 在机器学习领域，这种类型的预测被称为**多类分类**。

## 问题
这个问题集中在根据花瓣长度，花瓣宽度等花的参数预测鸢尾花（setosa，versicolor或virginica）的类型。

为了解决这个问题，我们将建立一个ML模型，它有4个输入参数： 
* petal length
* petal width
* sepal length
* sepal width

并预测该花属于哪种鸢尾花类型：
* setosa
* versicolor
* virginica

确切地说，模型将返回花属于每个类型的概率。

## ML 任务 - 多类分类
**多类分类**的广义问题是将项目分类为三个或更多类别中的一个。 （将项目分类为两个类别之一称为**二元分类**）。

多类分类的其他例子包括： 
* 手写数字识别：预测图像中包含10个数字（0～9）。
* 问题标记：预测问题属于哪个类别（UI，后端，文档）。
* 根据患者的测试结果预测疾病阶段。

所有这些例子的共同特点是我们要预测的参数可以取几个（超过两个）值中的一个。换句话说，这个值由`enum`表示，而不是由`integer`、`float`、`double`或`boolean`类型表示。

## 解决方案
为了解决这个问题，首先我们将建立一个ML模型。然后，我们将在现有数据上训练模型，评估其有多好，最后我们将使用该模型来预测鸢尾花类型。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)

### 1. 建立模型

建立模型包括: 
* 使用`DataReader`上传数据（`iris-train.txt`）
* 创建一个评估器并将数据转换为一列，以便ML算法（使用`Concatenate`）可以有效地使用它。
* 选择学习算法（`StochasticDualCoordinateAscent`）。


初始代码类似以下内容： 
```CSharp
// Create MLContext to be shared across the model creation workflow objects 
// Set a random seed for repeatable/deterministic results across multiple trainings.
var mlContext = new MLContext(seed: 0);

// STEP 1: Common data loading configuration
var trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
var testDataView = mlContext.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true);

// STEP 2: Common data process configuration with pipeline data transformations
var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "KeyColumn", inputColumnName: nameof(IrisData.Label))
        .Append(mlContext.Transforms.Concatenate("Features", nameof(IrisData.SepalLength),
                                                            nameof(IrisData.SepalWidth),
                                                            nameof(IrisData.PetalLength),
                                                            nameof(IrisData.PetalWidth))
                                                            .AppendCacheCheckpoint(mlContext)); 
                                                            // Use in-memory cache for small/medium datasets to lower training time. 
                                                            // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets. 


// STEP 3: Set the training algorithm, then create and config the modelBuilder                         
var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "KeyColumn", featureColumnName: "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: nameof(IrisData.Label) , inputColumnName: "KeyColumn"));

var trainingPipeline = dataProcessPipeline.Append(trainer);
```

### 2. 训练
训练模型是在训练数据（已知鸢尾花类型）上运行所选算法以调整模型参数的过程。它在评估器对象中的`Fit()` 方法中实现。 

为了执行训练，我们只需调用方法时传入在DataView对象中提供的训练数据集（iris-train.txt文件）。

```CSharp
// STEP 4: Train the model fitting to the DataSet            

ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);

```
### 3. 评估模型
我们需要这一步来总结我们的模型对新数据的准确性。 为此，上一步中的模型针对另一个未在训练中使用的数据集（`iris-test.txt`）运行。 此数据集还包含已知的鸢尾花类型。 `MulticlassClassification.Evaluate`在各种指标中计算模型预测的值和已知类型之间的差异。

```CSharp
var predictions = trainedModel.Transform(testDataView);
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");

Common.ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
```

>*要了解关于如何理解指标的更多信息，请参阅[ML.NET指南](https://docs.microsoft.com/en-us/dotnet/machine-learning/) 中的机器学习词汇表，或者使用任何有关数据科学和机器学习的可用材料*.

如果您对模型的质量不满意，可以采用多种方法来改进，这将在*examples*类别中进行介绍。 
### 4. 使用模型
在模型被训练之后，我们可以使用`Predict()` API来预测这种花属于每个鸢尾花类型的概率。

```CSharp

ITransformer trainedModel;
using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
{
    trainedModel = mlContext.Model.Load(stream);
}

// Create prediction engine related to the loaded trained model
var predEngine = trainedModel.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext);

// During prediction we will get Score column with 3 float values.
// We need to find way to map each score to original label.
// In order to do what we need to get TrainingLabelValues from Score column.
// TrainingLabelValues on top of Score column represent original labels for i-th value in Score array.
// Let's look how we can convert key value for PredictedLabel to original labels.
// We need to read KeyValues for "PredictedLabel" column.
VBuffer<float> keys = default;
predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
var labelsArray = keys.DenseValues().ToArray();
// Since we apply MapValueToKey estimator with default parameters, key values
// depends on order of occurence in data file. Which is "Iris-setosa", "Iris-versicolor", "Iris-virginica"
// So if we have Score column equal to [0.2, 0.3, 0.5] that's mean what score for
// Iris-setosa is 0.2
// Iris-versicolor is 0.3
// Iris-virginica is 0.5.
//Add a dictionary to map the above float values to strings. 
Dictionary<float, string> IrisFlowers = new Dictionary<float, string>();
IrisFlowers.Add(0, "Setosa");
IrisFlowers.Add(1, "versicolor");
IrisFlowers.Add(2, "virginica");

Console.WriteLine("=====Predicting using model====");
//Score sample 1
var resultprediction1 = predEngine.Predict(SampleIrisData.Iris1);

Console.WriteLine($"Actual: setosa.     Predicted label and score: {IrisFlowers[labelsArray[0]]}:      {resultprediction1.Score[0]:0.####}");
Console.WriteLine($"                                           {IrisFlowers[labelsArray[1]]}:  {resultprediction1.Score[1]:0.####}"); Console.WriteLine($"                                           {IrisFlowers[labelsArray[2]]}:   {resultprediction1.Score[2]:0.####}");
Console.WriteLine();
```

`TestIrisData.Iris1`中存储着有关我们想要预测类型的花的信息。

```CSharp
internal class TestIrisData
{
    internal static readonly IrisData Iris1 = new IrisData()
    {
        SepalLength = 3.3f,
        SepalWidth = 1.6f,
        PetalLength = 0.2f,
        PetalWidth= 5.1f,
    }
    (...)
}
```
