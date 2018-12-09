# eShopDashboardML - 销售预测 

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v0.7           | 动态 API | 最新版本 | ASP.NET Core Web应用程序和控制台应用程序 | SQL Server 和 .csv 文件 | 销售预测  | 回归 | FastTreeTweedie 回归 |


eShopDashboardML是一个使用[ML.NET](https://github.com/dotnet/machinelearning) 进行（每个产品和每个地区）销售预测的Web应用程序。


# 概述

这个终端示例应用程序通过展现以下主题着重介绍ML.NET API的用法:

1. 如何训练，建立和生成ML模型
   - 使用.NET Core实现一个[控制台应用程序](src\eShopForecastModelsTrainer)。
2. 如何使用经过训练的ML模型做下个月的销售预测
   - 使用[ASP.NET Core Razor](https://docs.microsoft.com/aspnet/core/tutorials/razor-pages/)实现一个独立的，单体[Web应用程序](src\eShopDashboard)。

该应用程序还使用一个SQL Server数据库存储常规产品目录和订单信息，就像许多使用SQL Server的典型Web应用程序一样。在本例中，由于它是一个示例，因此默认情况下使用localdb SQL数据库，因此不需要设置真正的SQL Server。在第一次运行Web应用程序时，将创建localdb数据库并包含示例数据。

如果要使用真正的SQL Server或Azure SQL数据库，只需更改应用程序中的连接字符串即可。

这是Web应用程序的一个销售预测屏幕截图示例：

![image](./docs/images/eShopDashboard.png)

## 演练：如何设置

了解如何在 Visual Studio 中设置以及对代码的进一步说明：

- [在 Visual Studio 中设置 eShopDashboard 并运行Web应用程序](docs/Setting-up-eShopDashboard-in-Visual-Studio-and-running-it.md)

- [创建和训练您的ML模型](docs/Create-and-train-the-models-%5BOptional%5D.md)
  - 此步骤是可选的，因为Web应用程序已配置为使用预先训练的模型。 但是，您可以创建自己的训练模型，并将预先训练的模型与您自己的模型交换。

## 演练：ML.NET代码实现

### 问题

这个问题是基于之前的销售情况围绕地区和产品进行销售预测

### 数据集

为了解决这个问题，您建立了两个独立的ML模型，它们以以下数据集作为输入：

| 数据集 | 列 |
|----------|--------|
| **products stats**  | next, productId, year, month, units, avg, count, max, min, prev      |
| **country stats**  | next, country, year, month, max, min, std, count, sales, med, prev   |

### ML 任务 - [回归](https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/tasks#regression)

这个示例的ML任务是回归，它是一个有监督的机器学习任务，用于从一组相关的特征/变量中预测下一个周期的值（在本例中是销售预测）。

### 解决方案

为了解决这个问题，首先我们将建立ML模型，同时根据现有数据训练每个模型，评估其有多好，最后使用模型预测销售。

注意，该示例实现了两个独立的模型：
- 下一个周期（月）产品需求预测模型
- 下一个周期（月）地区销售预测模型

当然，当学习/研究此示例时，您可以只关注其中一个场景/模型。

![建立 -> 训练 -> 评估 -> 使用](docs/images/modelpipeline.png)

#### 1. 建立模型

您需要实现的第一步是定义要从数据集文件加载的数据列，如下面的代码所示：

[建立并训练模型](./src/eShopForecastModelsTrainer/ProductModelHelper.cs)

```csharp
var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments
                        {
                            Column = new[] {
                                new TextLoader.Column("next", DataKind.R4, 0 ),
                                new TextLoader.Column("productId", DataKind.Text, 1 ),
                                new TextLoader.Column("year", DataKind.R4, 2 ),
                                new TextLoader.Column("month", DataKind.R4, 3 ),
                                new TextLoader.Column("units", DataKind.R4, 4 ),
                                new TextLoader.Column("avg", DataKind.R4, 5 ),
                                new TextLoader.Column("count", DataKind.R4, 6 ),
                                new TextLoader.Column("max", DataKind.R4, 7 ),
                                new TextLoader.Column("min", DataKind.R4, 8 ),
                                new TextLoader.Column("prev", DataKind.R4, 9 )
                            },
                            HasHeader = true,
                            Separator = ","
                        });
```

然后，下一步是构建转换管道，并指定要使用什么训练器/算法。
在这个案例中，您将进行以下转换：
- 连接当前特征生成名为NumFeatures的新列
- 使用[独热编码](https://en.wikipedia.org/wiki/One-hot)转换productId
- 连接所有生成的特征生成名为'Features'的新列
- 复制“next”列将其重命名为“Label”
- 指定“Fast Tree Tweedie”训练器作为算法应用于模型

在设计管道之后，您可以将数据集加载到DataView中，而且此步骤只是配置，DataView是延迟加载，在下一步训练模型之前数据不会被加载。

```csharp
var trainingPipeline = mlContext.Transforms.Concatenate(outputColumn: "NumFeatures", "year", "month", "units", "avg", "count", "max", "min", "prev" )
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(inputColumn:"productId", outputColumn:"CatFeatures"))
    .Append(mlContext.Transforms.Concatenate(outputColumn: "Features", "NumFeatures", "CatFeatures"))
    .Append(mlContext.Transforms.CopyColumns("next", "Label"))
    .Append(trainer = mlContext.Regression.Trainers.FastTreeTweedie("Label", "Features"));

var trainingDataView = textLoader.Read(dataPath);
```


#### 2. 训练模型

在建立管道之后，我们通过使用所选算法拟合或使用训练数据来训练预测模型。 在该步骤中，模型被建立，训练并作为对象返回：

```csharp
var model = trainingPipeline.Fit(trainingDataView);
```

#### 3. 评估模型

在本例中，模型的评估是在使用交叉验证方法训练模型之前执行的，因此您将获得指示模型准确度的指标。

```csharp
var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, numFolds: 6, labelColumn: "Label");
            
ConsoleHelper.PrintRegressionFoldsAverageMetrics(trainer.ToString(), crossValidationResults);
```

#### 4. 保存模型供最终用户的应用程序稍后使用

一旦创建和评估了模型，就可以将它保存到.ZIP文件中，任何最终用户的应用程序都可以通过以下代码使用它：

```csharp            
using (var file = File.OpenWrite(outputModelPath))
    model.SaveTo(mlContext, file);
```

#### 5. 用简单的测试预测试用模型

简单地说，您可以从.ZIP文件中加载模型，创建一些示例数据，创建“预测函数”，最后进行预测。 

```csharp
ITransformer trainedModel;
using (var stream = File.OpenRead(outputModelPath))
{
    trainedModel = mlContext.Model.Load(stream);
}

var predictionFunct = trainedModel.MakePredictionFunction<ProductData, ProductUnitPrediction>(mlContext);

Console.WriteLine("** Testing Product 1 **");

// Build sample data
ProductData dataSample = new ProductData()
{
    productId = "263",
    month = 10,
    year = 2017,
    avg = 91,
    max = 370,
    min = 1,
    count = 10,
    prev = 1675,
    units = 910
};

//model.Predict() predicts the nextperiod/month forecast to the one provided
ProductUnitPrediction prediction = predictionFunct.Predict(dataSample);
Console.WriteLine($"Product: {dataSample.productId}, month: {dataSample.month + 1}, year: {dataSample.year} - Real value (units): 551, Forecast Prediction (units): {prediction.Score}");

```

## 引用
eShopDashboardML数据集是基于**UCI**(http://archive.ics.uci.edu/ml/datasets/online+retail) 的一个公共在线零售数据集
> Daqing Chen, Sai Liang Sain, 和 Kun Guo, 在线零售业的数据挖掘: 基于RFM模型的数据挖掘客户细分案例研究, 数据库营销与客户战略管理杂志, Vol. 19, No. 3, pp. 197â€“208, 2012 (印刷前在线发布: 27 August 2012. doi: 10.1057/dbm.2012.17).
