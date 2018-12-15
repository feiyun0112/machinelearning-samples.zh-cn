# 聚类鸢尾花数据

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v0.7           | 动态 API | 最新版 | 控制台应用程序 | .txt 文件 | 聚类鸢尾花 | 聚类 | K-means++ |

在这个介绍性示例中，您将看到如何使用[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)将不同类型鸢尾花划分为不同组。在机器学习的世界中，这个任务被称为**群集**。

## 问题
为了演示聚类API的实际作用，我们将使用三种类型的鸢尾花：setosa、versicolor和versicolor。它们都存储在相同的数据集中。尽管这些花的类型是已知的，我们将不使用它，只对花的参数，如花瓣长度，花瓣宽度等运行聚类算法。这个任务是把所有的花分成三个不同的簇。我们期望不同类型的花属于不同的簇。

模型的输入使用下列鸢尾花参数：
* petal length
* petal width
* sepal length
* sepal width

## ML 任务 - 聚类
**聚类**的一般问题是将一组对象分组，使得同一组中的对象彼此之间的相似性大于其他组中的对象。

其他一些聚类示例：
* 将新闻文章分为不同主题：体育，政治，科技等。
* 按购买偏好对客户进行分组。
* 将数字图像划分为不同的区域以进行边界检测或物体识别。

聚类看起来类似于多类分类，但区别在于对于聚类任务，我们不知道过去数据的答案。 因此，没有“导师”/“主管”可以判断我们的算法的预测是对还是错。 这种类型的ML任务称为**无监督学习**。

## 解决方案
要解决这个问题，首先我们将建立并训练ML模型。 然后我们将使用训练模型来预测鸢尾花的簇。

### 1. 建立模型

建立模型包括：上传数据（使用`TextLoader`加载`iris-full.txt`），转换数据以便ML算法（使用`Concatenate`）有效地使用，并选择学习算法（`KMeans`）。 所有这些步骤都存储在`trainingPipeline`中：
```CSharp
//Create the MLContext to share across components for deterministic results
MLContext mlContext = new MLContext(seed: 1);  //Seed set to any number so you have a deterministic environment

// STEP 1: Common data loading configuration
TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
                                {
                                    Separator = "\t",
                                    HasHeader = true,
                                    Column = new[]
                                                {
                                                    new TextLoader.Column("Label", DataKind.R4, 0),
                                                    new TextLoader.Column("SepalLength", DataKind.R4, 1),
                                                    new TextLoader.Column("SepalWidth", DataKind.R4, 2),
                                                    new TextLoader.Column("PetalLength", DataKind.R4, 3),
                                                    new TextLoader.Column("PetalWidth", DataKind.R4, 4),
                                                }
                                });

IDataView fullData = textLoader.Read(DataPath);

//STEP 2: Process data transformations in pipeline
var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth");

// STEP 3: Create and train the model     
var trainer = mlContext.Clustering.Trainers.KMeans(features: "Features", clustersCount: 3);
var trainingPipeline = dataProcessPipeline.Append(trainer);
```
### 2. 训练模型
训练模型是在给定数据上运行所选算法的过程。 要执行训练，您需要调用`Fit()`方法。
```CSharp
var trainedModel = trainingPipeline.Fit(trainingDataView);
```
### 3. 使用模型
在建立和训练模型之后，我们可以使用`Predict()`API来预测鸢尾花的簇，并计算从给定花参数到每个簇（簇的每个质心）的距离。

```CSharp
                // Test with one sample text 
                var sampleIrisData = new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f,
                };

                // Create prediction engine related to the loaded trained model
                var predFunction = trainedModel.MakePredictionFunction<IrisData, IrisPrediction>(mlContext);

                //Score
                var resultprediction = predFunction.Predict(sampleIrisData);
                
                Console.WriteLine($"Cluster assigned for setosa flowers:" + resultprediction.SelectedClusterId);
```