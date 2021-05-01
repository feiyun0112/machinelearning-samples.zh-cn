# 电影推荐 - 矩阵分解示例

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| Microsoft.ML.Recommender Preview v0.16.0   | 动态 API | 最新版本 | 控制台应用程序| .csv 文件 | 推荐 | 矩阵分解 | MatrixFactorizationTrainer|

在这个示例中，您可以看到如何使用ML.NET来构建电影推荐引擎。


## 问题
在本教程中，我们将使用MovieLens数据集，其中包含电影评分，标题，流派等信息。在构建我们的电影推荐引擎的方法方面，我们将使用分解机，它使用协同过滤方法。

“协同过滤”是在一个基本假设的情况下运作的，即如果某人A在一个问题上与某人B具有相同的意见，则在另一个问题上，相对其他随机选择的人，A更倾向于B的观点。

使用ML.NET，我们支持以下三种推荐场景，根据您的场景，您可以从下面的列表中选择三种场景之一。

| 场景 | 算法 | 示例链接
| --- | --- | --- | 
| 你有用户购买行为中的用户Id、产品Id和评分。| 矩阵分解 | 当前示例 | 
| 你仅有用户购买行为中用户Id和产品Id，但是没有评分。 这在来自在线商店的数据集中很常见，您可能只能访问客户的购买历史记录。 有了这种类型的推荐，你可以建立一个推荐引擎用来推荐经常购买的物品。| One Class 矩阵分解 | [产品推荐器](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/MatrixFactorization_ProductRecommendation) | 
| 您希望在您的推荐引擎中使用用户Id、产品Id和评分之外的更多属性（特征），例如产品描述，产品价格等。 | 场感知分解机  | [基于分解机的电影推荐器](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/end-to-end-apps/Recommendation-MovieRecommender/MovieRecommender_Model) | 


## DataSet
原始数据来自MovieLens数据集：
http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

## ML 任务 - [矩阵分解（推荐）](https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/tasks#recommendation)

这个示例的ML任务是矩阵分解，它是一个执行协同过滤的有监督的机器学习任务。

## 解决方案

要解决此问题，您需要在现有训练数据上建立和训练ML模型，评估其有多好（分析获得的指标），最后您可以使用/测试模型来预测给定输入数据变量的需求。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)

### 1. 建立模型

建立模型包括: 

* 定义映射到数据集的数据模式，并使用DataReader读取（`recommended-ratings-train.csv`和`recommended-ratings-test.csv`）

* 矩阵分解需要对userId，movieId这两个特征进行编码

* 然后MatrixFactorizationTrainer将这两个已编码特征（userId, movieId）作为输入

下面是用于建立模型的代码：
```CSharp
 
 //STEP 1: Create MLContext to be shared across the model creation workflow objects 
  MLContext mlcontext = new MLContext();

 //STEP 2: Read the training data which will be used to train the movie recommendation model    
 //The schema for training data is defined by type 'TInput' in LoadFromTextFile<TInput>() method.
 IDataView trainingDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TrainingDataLocation, hasHeader: true, ar:',');

//STEP 3: Transform your data by encoding the two features userId and movieID. These encoded features will be provided as 
//        to our MatrixFactorizationTrainer.
 var dataProcessingPipeline = mlcontext.Transforms.Conversion.MapValueToKey(outputColumnName: userIdEncoded, inputColumnName: eRating.userId))
                .Append(mlcontext.Transforms.Conversion.MapValueToKey(outputColumnName: movieIdEncoded, inputColumnName: nameofg.movieId)));
 
 //Specify the options for MatrixFactorization trainer
 MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
 options.MatrixColumnIndexColumnName = userIdEncoded;
 options.MatrixRowIndexColumnName = movieIdEncoded;
 options.LabelColumnName = "Label";
 options.NumberOfIterations = 20;
 options.ApproximationRank = 100;

//STEP 4: Create the training pipeline 
 var trainingPipeLine = dataProcessingPipeline.Append(mlcontext.Recommendation().Trainers.MatrixFactorization(options));

```


### 2. 训练模型
训练模型是在训练数据（具有已知电影和用户评分）上运行所选算法以调整模型参数的过程。 它是在评估器对象的`Fit()`方法中实现的。

要执行训练，您需要调用`Fit()`方法访问在DataView对象中提供的训练数据集（`recommendation-ratings-train.csv`文件）。

```CSharp    
ITransformer model = trainingPipeLine.Fit(trainingDataView);
```
请注意，ML.NET使用延迟加载方法处理数据，所以实际上只有调用.Fit()方法时才真正在内存中加载数据。

### 3. 评估模型
我们需要这一步来总结我们的模型对新数据的准确性。 为此，上一步中的模型针对未在训练中使用的另一个数据集运行（`recommendation-ratings-test.csv`）。

`Evaluate()` 比较测试数据集的预测值并生成各种指标，例如准确性，您可以进行研究。

```CSharp 
Console.WriteLine("=============== Evaluating the model ===============");
IDataView testDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TestDataLocation, hasHeader: true); 
var prediction = model.Transform(testDataView);
var metrics = mlcontext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
```

### 4. 使用模型
训练模型后，您可以使用`Predict()`API来预测特定电影/用户组合的评分。
```CSharp    
var predictionengine = mlcontext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
var movieratingprediction = predictionengine.Predict(
                new MovieRating()
                {
                    //Example rating prediction for userId = 6, movieId = 10 (GoldenEye)
                    userId = predictionuserId,
                    movieId = predictionmovieId
                }
            );
 Console.WriteLine("For userId:" + predictionuserId + " movie rating prediction (1 - 5 stars) for movie:" +  
                   movieService.Get(predictionmovieId).movieTitle + " is:" + Math.Round(movieratingprediction.Score,1));
       
```
方案，我们也将为其建立示例。

#### 矩阵分解的得分

矩阵分解产生的分数表示为正的可能性。得分值越大，成为阳性案例的概率越高。然而，分数没有任何概率信息。当你做一个预测时，你必须计算出多个商品的得分，并挑选得分最高的商品。
