# 将ML.NET模型导出到ONNX

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.5.5           | 动态API | 最新 | 控制台应用程序 | .csv文件 | 价格预测 | 回归  | Light GBM regression |

在这个示例中，您将看到如何使用ML.NET来训练回归模型，然后将该模型转换为ONNX格式。

## 问题

开放式神经网络交换即[ONNX](http://onnx.ai/)是一种表示深度学习模型的开放格式。使用ONNX，开发人员可以在最先进的工具之间移动模型，并选择最适合他们的组合。ONNX是由一个合作伙伴社区开发和支持的。

有时您可能希望使用ML.NET训练模型，然后转换为ONNX，例如，如果您希望使用WinML使用模型以利用Windows应用程序中的GPU推断。

不是所有的ML.NET模型都可以转换成ONNX；它依赖于训练器和训练管道中的变换。有关支持的训练器列表，请参见ML.NET[Algorithms Doc](https://docs.microsoft.com/dotnet/machine-learning/how-to-choose-an-ml-net-algorithm)中的表格，有关支持的转换的列表请查看[Data transforms Doc](https://docs.microsoft.com/dotnet/machine-learning/resources/transforms)。
## Dataset

本示例使用[NYC出租车票价数据集](https://github.com/dotnet/machinelearning-samples/blob/main/datasets/README.md#nyc-taxi-fare)。

## 数据集

控制台应用程序项目`ONNXExport` 用于训练一个ML.NET模型，该模型根据行驶距离和乘客数量等特征预测出租车票价，将该模型导出到ONNX，然后使用ONNX模型进行预测。

### NuGet包

要将ML.NET模型导出到ONNX，必须在项目中安装以下NuGet包：

- Microsoft.ML.OnnxConverter

您还必须安装：

- Microsoft.ML, 用于训练ML.NET模型
- Microsoft.ML.ONNXRuntime和Microsoft.ML.OnnxTransformer，用于为ONNX模型评分

### 转换和训练器

此管道包含以下转换和训练器，它们都是ONNX可导出的：

- OneHotEncoding 转换
- Concatenate 转换
- Light GBM 训练器

### 代码

训练ML.NET模型后，可以使用以下代码转换为ONNX：

```csharp
using (var stream = File.Create("taxi-fare-model.onnx"))
   mlContext.Model.ConvertToOnnx(model, trainingDataView, stream);
```

您需要一个转换器和输入数据来将ML.NET模型转换为ONNX模型。默认情况下，ONNX转换将生成具有最新OpSet版本的ONNX文件

转换为ONNX后，可以使用以下代码使用ONNX模型：

```csharp
var onnxEstimator = mlContext.Transforms.ApplyOnnxModel(onnxModelPath);

using var onnxTransformer = onnxEstimator.Fit(trainingDataView);

var onnxOutput = onnxTransformer.Transform(testDataView);
```

在同一个示例输入上比较ML.NET模型和ONNX模型时，应该会得到相同的结果。如果运行项目，则应在控制台中获得类似于以下输出的结果：

```console
Predicted Scores with ML.NET model
Score      19.60645
Score      18.673796
Score      5.9175444
Score      4.8969507
Score      19.108932
Predicted Scores with ONNX model
Score      19.60645
Score      18.673796
Score      5.9175444
Score      4.8969507
Score      19.108932
```

## 性能

默认的ONNX到ML.NET的转换不是最佳的，并且会产生ML.NET使用不需要的额外图形输出。ONNX运行时执行反向深度优先搜索，这会导致大量从ONNX运行时到ML.NET的本机内存到托管内存的转换操作，并执行超过必要内核的操作。

如果只指定必要的图形输出，它将只执行图形的一个子集。因此，通过消除除Score之外的所有图形输出，可以提高推理性能。
