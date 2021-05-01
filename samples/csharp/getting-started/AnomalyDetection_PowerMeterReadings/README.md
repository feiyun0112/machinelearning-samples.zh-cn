# 功耗异常检测

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4         | 动态 API | 最新版 | 控制台应用程序 | .csv 文件 | 电表异常检测| 时间序列 - 异常检测| SsaSpikeDetection |

在这个示例中，您将看到如何使用[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)来检测时间序列数据中的异常。

## 问题
这个问题的重点是根据智能电表的每日读数来找出用电量的峰值。

为了解决这个问题，我们将使用下列输入构建一个ML模型：
* 日期和时间
* 仪表读数差，通过读数之间的时间跨度进行标准化（ConsumptionDiffNormalized）

并在检测到异常时生成警报。

## ML 任务 - 时间序列
目标是识别罕见的项目、事件或观察结果，这些项目、事件或观察结果与大多数时间序列数据存在显著差异，从而引起怀疑。

## 解决方案
要解决此问题，您需要在现有训练数据上构建和训练ML模型，评估其有多好（分析获得的指标），最后您可以使用/测试模型来预测给定输入数据变量的需求。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)

然而，在这个例子中，我们将建立和训练模型来演示时间序列异常检测库，因为它使用实际数据检测，并且没有评估方法。然后，我们将在预测输出列中查看检测到的异常。

### 1. 建立模型
建立模型包括：

- 使用LoadFromTextFile准备和加载数据

- 时间序列评估器的选择与参数设置


初始代码类似于以下代码：

`````csharp

// Create a common ML.NET context.
var ml = new MLContext();

[...]

// Create a class for the dataset
class MeterData
{
    [LoadColumn(0)]
    public string name { get; set; }
    [LoadColumn(1)]
    public DateTime time { get; set; }
    [LoadColumn(2)]
    public float ConsumptionDiffNormalized { get; set; }
}

[...]

// Load the data
[...]

var dataView = ml.Data.LoadFromTextFile<MeterData>(
                TrainingData,
                separatorChar: ',',
                hasHeader: true);

[...]

// Prepare the Prediction output column for the model
class SpikePrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; }
}

[...]

// Configure the Estimator
const int PValueSize = 30;
const int SeasonalitySize = 30;
const int TrainingSize = 90;
const int ConfidenceInterval = 98;

string outputColumnName = nameof(SpikePrediction.Prediction);
string inputColumnName = nameof(MeterData.ConsumptionDiffNormalized);  

var trainigPipeLine = mlContext.Transforms.DetectSpikeBySsa(
                outputColumnName,
                inputColumnName,
                confidence: ConfidenceInterval,
                pvalueHistoryLength: PValueSize,
                trainingWindowSize: TrainingSize,
                seasonalityWindowSize: SeasonalitySize);

`````

### 2. 训练模型
训练模型是在训练数据（具有已知异常值）上运行所选算法以调整模型参数的过程。它是在Estimator对象的`Fit()`方法中实现的。

要执行训练，需要在DataView对象中提供训练数据集（`power-export_min.csv`）时调用`Fit()`方法。

`````csharp    
ITransformer trainedModel = trainigPipeLine.Fit(dataView);
`````

### 3. 查看异常
通过访问输出列，可以查看时间序列模型中检测到的异常。

`````csharp    
var transformedData = model.Transform(dataView);
`````
