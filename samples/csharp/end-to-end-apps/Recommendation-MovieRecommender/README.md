# 电影推荐 

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
|v1.4     | 动态 API | 最新版本 | 终端应用程序 | .csv | 电影推荐 | 推荐 | Field Aware Factorization Machines |

![Alt Text](https://github.com/dotnet/machinelearning-samples/blob/master/samples/csharp/end-to-end-apps/Recommendation-MovieRecommender/MovieRecommender/movierecommender/wwwroot/images/movierecommender.gif)

## 概述

MovieRecommender是一个简单的应用程序，它构建和使用推荐模型。

这是一个关于如何使用推荐来增强现有ASP.NET应用程序的终端示例。

本示例从流行的Netflix应用程序中汲取了灵感，并且尽管这个示例主要关注电影推荐，但是可以很容易地应用于任何类型的产品推荐。

## 特点
* Web应用程序 
    * 这是一个终端ASP.NET应用程序，它包含了三个用户'Ankit'，'Cesar'，'Gal'。然后，它使用ML.NET推荐模型给这三个用户提供建议。
      
* 推荐模型 
    * 应用程序使用MovieLens数据集构建推荐模型。模型训练代码使用基于协同过滤的推荐方法。

## 它如何工作？

## 训练模型

Movie Recommender 使用基于协同过滤的推荐方法。

协同过滤的基本假设是，如果A（例如Gal）在某个问题上与B（例如Cesar）具有相同的观点，则A（Gal）更有可能在另一个问题上具有和B（Cesar）相同的意见，而不是一个随机的人。

对于本示例，我们使用 http://files.grouplens.org/datasets/movielens/ml-latest-small.zip 数据集。

模型训练代码可以在[MovieRecommender_Model](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/end-to-end-apps/Recommendation-MovieRecommender/MovieRecommender_Model)中找到。

模型训练遵循以下四个步骤来构建模型。 您可以先跳过代码并继续。

![Build -> Train -> Evaluate -> Consume](https://github.com/dotnet/machinelearning-samples/blob/master/samples/csharp/getting-started/shared_content/modelpipeline.png)

## 使用模型

通过以下步骤在[Controller](https://github.com/dotnet/machinelearning-samples/blob/master/samples/csharp/end-to-end-apps/Recommendation-MovieRecommender/MovieRecommender/movierecommender/Controllers/MoviesController.cs#L60)中使用训练的模型。

### 1. 创建ML.NET环境并加载已经训练过的模型

```CSharp

   // 1. Create the ML.NET environment and load the already trained model
   MLContext mlContext = new MLContext();
            
   ITransformer trainedModel;
   using (var stream = new FileStream(_movieService.GetModelPath(), FileMode.Open, FileAccess.Read, FileShare.Read))
   {
    trainedModel = mlContext.Model.Load(stream);
   }
 ```
### 2. 创建预测函数以预测一组电影推荐

```CSharp
   //2. Create a prediction function
   var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(trainedModel);
            
   List<(int movieId, float normalizedScore)> ratings = new List<(int movieId, float normalizedScore)>();
   var MovieRatings = _profileService.GetProfileWatchedMovies(id);
   List<Movie> WatchedMovies = new List<Movie>();

   foreach ((int movieId, int movieRating) in MovieRatings)
   {
     WatchedMovies.Add(_movieService.Get(movieId));
   }
   
   MovieRatingPrediction prediction = null;
   
   foreach (var movie in _movieService.GetTrendingMovies)
   {
       //Call the Rating Prediction for each movie prediction
        prediction = predictionEngine.Predict(new MovieRating
        {
            userId = id.ToString(),
            movieId = movie.MovieID.ToString()
        });
       //Normalize the prediction scores for the "ratings" b/w 0 - 100
       float normalizedscore = Sigmoid(prediction.Score);
       //Add the score for recommendation of each movie in the trending movie list
        ratings.Add((movie.MovieID, normalizedscore));
   }
 ```

### 3. 为要显示的视图提供评分预测

```CSharp
  ViewData["watchedmovies"] = WatchedMovies;
  ViewData["ratings"] = ratings;
  ViewData["trendingmovies"] = _movieService.GetTrendingMovies;
  return View(activeprofile);
 ```

## 替代方法 
这个示例显示了许多可以用于ML.NET的推荐方法之一。根据您的特定场景，您可以选择以下任何最适合您的用例的方法。

| 场景 | 算法 | 示例链接
| --- | --- | --- | 
| 您想使用诸如用户Id、产品Id、评分、产品描述、产品价格等属性（特性）作为推荐引擎。在这种场景中，场感知分解机是一种通用的方法，您可以使用它来构建推荐引擎 | 场感知分解机 | 当前示例 | 
| 你有用用户购买行为中的户ID，产品和评分。对于这种情况，您应该使用矩阵分解法  | 矩阵分解 | [矩阵分解 - 推荐](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/MatrixFactorization_MovieRecommendation)| 
| 你仅有用户购买行为中用户Id和产品Id，但是没有评分。 这在来自在线商店的数据集中很常见，您可能只能访问客户的购买历史记录。 有了这种类型的推荐，你可以建立一个推荐引擎用来推荐经常购买的物品。 | One Class 矩阵分解 | [Product Recommender](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/MatrixFactorization_ProductRecommendation) | 



