# 手写数字识别

ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4           | 动态 API | 最新版本 | 控制台应用程序 | .csv 文件 | MNIST classification | 多类分类 | Sdca Multi-class |

在这个介绍性示例中，您将看到如何使用[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)通过MNIST数据集对从0到9的手写数字进行分类。这是一个**多类分类**问题，我们将使用SDCA（随机双坐标上升）算法来解决。

## 问题

MNIST数据集包含从0到9的手写数字图像。

我们使用的MNIST数据集包含65列数字。每行的前64列是0到16之间的整数值。通过将32 x 32位图划分为4 x 4的非重叠块来计算这些值。在这些块中的每个块中计算像素的数量，从而生成8 x 8的输入矩阵。每行的最后一列是由前64列中的值表示的数字。前64列是我们的特性，我们的ML模型将使用这些特性对测试图像进行分类。我们的训练和验证数据集中的最后一列是标签——我们将使用ML模型预测的实际数字。

我们将构建的ML模型将返回给定图像是0到9之间的数字之一的概率。

## 数据集

数据集可在UCI机器学习存储库获得，即 http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

[此处](./MNIST/Data/Datasets-Citation.txt)引用了该数据集

## ML 任务 - 多类分类
**多类分类**的广义问题是将项目分类为三个或更多类别中的一个。 （将项目分类为两个类别之一称为**二元分类**）。

## 解决方案
为了解决这个问题，首先我们将建立一个ML模型。然后，我们将在现有数据上训练模型，评估其有多好，最后我们将使用该模型来预测给定图像表示的数字。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)

### 1. 建立模型

建立模型包括: 
* 使用`DataReader`上传数据（`optdigits-train.csv`）
* 创建一个评估器并将并将前64列中的数据转换为一列，以便ML算法（使用`Concatenate`）可以有效地使用它。
* 选择学习算法（`StochasticDualCoordinateAscent`）。


初始代码类似以下内容： 
```CSharp
// STEP 1: Common data loading configuration
var trainData = mlContext.Data.LoadFromTextFile(path: TrainDataPath,
                        columns : new[] 
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader : false,
                        separatorChar : ','
                        );

                
var testData = mlContext.Data.LoadFromTextFile(path: TestDataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );

// STEP 2: Common data process configuration with pipeline data transformations
// Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.
var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Number").
                    Append(mlContext.Transforms.Concatenate("Features", nameof(InputData.PixelValues)).AppendCacheCheckpoint(mlContext));

// STEP 3: Set the training algorithm, then create and config the modelBuilder
var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("Number","Label"));
```

### 2. 训练模型
训练模型是在训练数据（已知鸢尾花类型）上运行所选算法以调整模型参数的过程。我们的训练数据由像素值和它们所代表的数字组成。它在评估器对象中的`Fit()` 方法中实现。 

为了执行训练，我们只需调用方法时传入在DataView对象中提供的训练数据集（optdigits-train.csv文件）。

```CSharp
// STEP 4: Train the model fitting to the DataSet            
ITransformer trainedModel = trainingPipeline.Fit(trainData);

```
### 3. 评估模型
我们需要这一步来总结我们的模型对新数据的准确性。 为此，上一步中的模型针对另一个未在训练中使用的数据集（`optdigits-val.csv`）运行。`MulticlassClassification.Evaluate`在各种指标中计算模型预测的值和已知类型之间的差异。

```CSharp
var predictions = trainedModel.Transform(testData);
var metrics = mlContext.MulticlassClassification.Evaluate(data:predictions, labelColumnName:"Number", scoreColumnName:"Score");

Common.ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
```

>*要了解关于如何理解指标的更多信息，请参阅[ML.NET指南](https://docs.microsoft.com/en-us/dotnet/machine-learning/) 中的机器学习词汇表，或者使用任何有关数据科学和机器学习的可用材料*.

如果您对模型的质量不满意，可以采用多种方法来改进，这将在*examples*类别中进行介绍。 

### 4. 使用模型
在模型被训练之后，我们可以使用`Predict()` API来预测正确数字的概率。

```CSharp

ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

// Create prediction engine related to the loaded trained model
var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(trainedModel);

var resultprediction1 = predEngine.Predict(SampleMNISTData.MNIST1);

Console.WriteLine($"Actual: 7     Predicted probability:       zero:  {resultprediction1.Score[0]:0.####}");
Console.WriteLine($"                                           One :  {resultprediction1.Score[1]:0.####}");
Console.WriteLine($"                                           two:   {resultprediction1.Score[2]:0.####}");
Console.WriteLine($"                                           three: {resultprediction1.Score[3]:0.####}");
Console.WriteLine($"                                           four:  {resultprediction1.Score[4]:0.####}");
Console.WriteLine($"                                           five:  {resultprediction1.Score[5]:0.####}");
Console.WriteLine($"                                           six:   {resultprediction1.Score[6]:0.####}");
Console.WriteLine($"                                           seven: {resultprediction1.Score[7]:0.####}");
Console.WriteLine($"                                           eight: {resultprediction1.Score[8]:0.####}");
Console.WriteLine($"                                           nine:  {resultprediction1.Score[9]:0.####}");
Console.WriteLine();

```

`SampleMNISTData.MNIST1`中存储着有关我们想要预测数字的像素值。

```CSharp
class SampleMNISTData
{
	internal static readonly InputData MNIST1 = new InputData()
	{
		PixelValues = new float[] { 0, 0, 0, 0, 14, 13, 1, 0, 0, 0, 0, 5, 16, 16, 2, 0, 0, 0, 0, 14, 16, 12, 0, 0, 0, 1, 10, 16, 16, 12, 0, 0, 0, 3, 12, 14, 16, 9, 0, 0, 0, 0, 0, 5, 16, 15, 0, 0, 0, 0, 0, 4, 16, 14, 0, 0, 0, 0, 0, 1, 13, 16, 1, 0 }
	}; //num 1
    (...)
}
```
