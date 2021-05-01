# 共享自行车需求 - 回归问题示例

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4 | 动态 API | 最新版 | 控制台应用程序 | .csv 文件 | 需求预测 | 回归 | Fast Tree regressor compared to additional regression algorithms|

在这个示例中，您可以看到如何使用ML.NET来预测自行车的需求。由于您试图基于过去的观测数据预测特定的数值，在机器学习中，这种类型的预测方法被称为回归。

## 问题

有关问题的更详细描述，请阅读原始文档中的详细信息 [
Bike Sharing Demand competition from Kaggle](https://www.kaggle.com/c/bike-sharing-demand).

## 数据集
原始数据来自公共UCI数据集:
https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset


## ML 任务 - [回归](https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/tasks#regression)

当前示例的ML任务是回归，它是一种受监督的机器学习任务，用于从一组相关的特征/变量中预测标签的值（在本例中是需求数量预测）。

## 解决方案

要解决此问题，您需要在现有训练数据上构建和训练ML模型，评估其有多好（分析获得的指标），最后您可以使用/测试模型来预测给定输入数据变量的需求。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)

然而，在这个例子中，我们训练多个模型（而不是单个模型），每个模型基于不同的回归学习器/算法，最后我们评估每个方法/算法的准确性，因此您可以更精确地选择训练模型。

以下列表是使用和比较的训练器/算法：

- Fast Tree
- Poisson Regressor
- SDCA (Stochastic Dual Coordinate Ascent) Regressor
- FastTreeTweedie

