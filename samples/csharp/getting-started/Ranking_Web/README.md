# 搜索引擎结果排名

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4         | 动态API       | 最新 |  控制台应用程序 | .csv 文件 | 搜索引擎结果排名 | 排名          | LightGBM |

这个介绍性示例演示如何使用ml.net来预测显示搜索引擎结果的最佳顺序。在机器学习领域，这种预测被称为排名。

## 问题
排名是搜索引擎面临的一个常见问题，因为用户希望查询结果根据其相关性进行排名/排序。这个问题包括各种业务场景，其中个性化排序是用户体验的关键，这已经超出了搜索引擎的需求。下面是几个具体的例子：
* 旅行社-提供一份酒店列表，列出最有可能被排名靠前的用户购买/预订的酒店。
* 购物-按与用户购物偏好一致的顺序显示产品目录中的项目。
* 招聘-检索根据最适合新职位空缺的候选人排列的职位申请。

排名对任何场景都很有用，在这些场景中，按照增加点击、购买、预订等可能性的顺序列出项目非常重要。
 
在这个示例中，我们展示了如何将排名应用于搜索引擎结果。要执行排名，目前有两种算法可用-FastTree Boosting（FastRank）和Light Gradient Boosting Machine（LightGBM）。在这个示例中，我们使用LightGBM的LambdaRank实现自动构建一个ML模型来预测排名。

## 数据集
本示例使用的数据基于最初由Microsoft Bing提供的公共[数据集](https://www.microsoft.com/en-us/research/project/mslr/)。数据集在[CC-by 4.0](https://creativecommons.org/licenses/by/4.0/)许可证下发布，包括训练、验证和测试数据。

```
@article{DBLP:journals/corr/QinL13,
  author    = {Tao Qin and 
               Tie{-}Yan Liu},
  title     = {Introducing {LETOR} 4.0 Datasets},
  journal   = {CoRR},
  volume    = {abs/1306.2597},
  year      = {2013},
  url       = {https://arxiv.org/abs/1306.2597},
  timestamp = {Mon, 01 Jul 2013 20:31:25 +0200},
  biburl    = {https://dblp.uni-trier.de/rec/bib/journals/corr/QinL13},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

此数据集的说明如下：

这些数据集是机器学习数据，其中查询和URL由ID表示。 数据集由从查询-网址对中提取的特征向量以及相关性判断标签组成：

* 相关性判断是从商业网络搜索引擎（Microsoft Bing）的一个失效标签集中获得的，该标签集取5个值，从0（不相关）到4（完全相关）。

* 这些特征基本上是由我们（微软）提取的，是研究界广泛使用的特征。

在这些数据文件中，每一行对应一个查询-网址对。第一列是相关标签，第二列是查询id，其他列是特性。相关性标签的值越大，查询-网址对的相关性就越大。查询-网址对由136维特征向量表示。

## 机器学习任务 - 排名
如前所述，本示例使用LightGBM LambdaRank算法，该算法使用名为[**Learning to Rank**](https://en.wikipedia.org/wiki/Learning_to_rank)的监督学习技术。该技术要求训练/验证/测试数据集包含一组数据实例，每个实例都用它们的相关性得分（例如相关性判断标签）进行标记。标签是一个数字/序数值，例如{0，1，2，3，4}。主题专家可以手动对这些数据实例及其相关度评分进行标记。或者，可以使用其他度量来确定标签，例如对给定搜索结果的单击次数。

预计数据集将具有比“完美”更多的“差的”相关性分数。这有助于避免将排名列表直接转换为大小相等的{0、1、2、3、4}容器。关联度得分也会被重用，这样**每个组**中大多数样本被标记为0，这意味着结果是“差的”。而只有一个或几个标记为4，这意味着结果是“完美”。下面是数据集标签分布的细分。您将注意到，0（差的）比4（完美）标签多70倍：
* Label 0 -- 624,263
* Label 1 -- 386,280
* Label 2 -- 159,451
* Label 3 -- 21,317
* Label 4 -- 8,881

一旦训练/验证/测试数据集被标记为相关分数，那么就可以使用这些数据对模型（ranker）进行训练和评估。通过模型训练过程，ranker学习如何根据标签值对组内的每个数据实例进行评分。单个数据实例的结果得分本身并不重要——相反，应该将这些得分相互比较，以确定组数据实例的相对顺序。一个数据实例的得分越高，它在其组中的相关性越强，排名也越高。

## 解决方案
由于本示例的数据集已经标记了相关分数，因此我们可以立即开始训练模型。如果您从一个没有标记的数据集开始，您将需要首先通过让主题专家提供相关性得分或使用其他一些度量来确定相关性来完成此过程。

通常，训练、验证和测试模型的模式包括以下步骤：
1. 模型是在**训练**数据集上训练的。然后使用**验证**数据集评估模型的度量。
2. 重复步骤1，通过重新训练和重新模型，直到达到所需的指标。此步骤的结果是应用必要的数据转换和训练器的管道。
3. 管道用于在组合的**训练**+**验证**数据集上训练。然后在**测试**数据集上评估模型的度量（仅一次）——这是用于测量模型质量的最后一组度量。
4. 最后一步是对组合的**训练**+**验证**+**测试**的**全部**数据集上重新训练管道。然后，该模型就可以部署到生产环境中了。

对模型在生产中的表现的最终估计是步骤3中的指标。生产环境使用的最终模型，根据所有可用数据进，在步骤4中进行训练。

本示例执行上述步骤的简化版本以对搜索引擎结果进行排序：
1. 管道是通过必要的数据转换和LightGBM LambdaRank训练器设置的。
2. 模型是使用**训练**数据集**训练**。然后使用**验证**数据集对模型进行**评估**。这将为每个搜索引擎结果生成一个**预测**。预测通过检查指标进行评估；特别是[Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)（NDCG）。
3. 管道用于使用**培训+验证**数据集**重新训练**模型。使用**测试**数据集对生成的模型进行**评估**——这是模型的最后一组度量。
4. 最后一次使用**培训+验证+测试**数据集对模型进行**再训练**。最后一步是**使用**模型来执行新传入搜索的排名预测。这将为每个搜索引擎结果产生一个**分数**。分数用于确定与同一查询中的其他结果（例如组）相关的排名。

### 1. 设置管道
本示例使用依赖于LightGBM LambdaRank的LightGbmRankingTrainer训练模型。模型需要以下输入列：

* Group Id—包含每个数据实例的组ID的列。数据实例包含在表示单个查询中所有候选结果的逻辑分组中，每个组都有一个称为组ID的标识符。对于搜索引擎数据集，搜索结果按其对应的查询分组，其中组ID对应于查询ID。组ID数据类型必须为[键类型](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.keydataviewtype)。
* Label—包含每个数据实例的相关性标签的列，其中较高的值表示较高的相关性。标签数据类型必须是[键类型](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.keydataviewtype)或[单精度浮点数](https://docs.microsoft.com/en-us/dotnet/api/system.single)。
* Features—确定数据实例的相关性/排名方面有影响的列。 特征数据必须是[单精度浮点数](https://docs.microsoft.com/en-us/dotnet/api/system.single)类型的固定大小向量。

当设置训练器时，**自定义增益**（或相关性增益）也可用于对每个标记的相关性得分应用权重。这有助于确保模型更加重视对权重更高的结果进行排名。在本示例中，我们使用默认提供的权重。

以下代码用于设置管道：

```CSharp
const string FeaturesVectorName = "Features";

// Load the training dataset.
IDataView trainData = mlContext.Data.LoadFromTextFile<SearchResultData>(trainDatasetPath, separatorChar: '\t', hasHeader: true);

// Specify the columns to include in the feature input data.
var featureCols = trainData.Schema.AsQueryable()
    .Select(s => s.Name)
    .Where(c =>
        c != nameof(SearchResultData.Label) &&
        c != nameof(SearchResultData.GroupId))
    .ToArray();

// Create an Estimator and transform the data:
// 1. Concatenate the feature columns into a single Features vector.
// 2. Create a key type for the label input data by using the value to key transform.
// 3. Create a key type for the group input data by using a hash transform.
IEstimator<ITransformer> dataPipeline = mlContext.Transforms.Concatenate(FeaturesVectorName, featureCols)
    .Append(mlContext.Transforms.Conversion.MapValueToKey(nameof(SearchResultData.Label)))
    .Append(mlContext.Transforms.Conversion.Hash(nameof(SearchResultData.GroupId), nameof(SearchResultData.GroupId), numberOfBits: 20));

// Set the LightGBM LambdaRank trainer.
IEstimator<ITransformer> trainer = mlContext.Ranking.Trainers.LightGbm(labelColumnName: nameof(SearchResultData.Label), featureColumnName: FeaturesVectorName, rowGroupColumnName: nameof(SearchResultData.GroupId));
IEstimator<ITransformer> trainerPipeline = dataPipeline.Append(trainer);
`````

### 2. 训练与评估模型
首先，我们需要使用**训练**数据集训练我们的模型。然后，我们需要评估我们的模型，以确定它在排名上的有效性。为此，模型将针对另一个未在训练中使用的数据集（**验证**数据集）运行。

`Evaluate()`将**验证**数据集的预测值与数据集的标签进行比较，并生成您可以探索的各种度量。具体来说，我们可以使用`Evaluate()`返回的`RankingMetrics`中包含的Discounted Cumulative Gain（DCG）和Normalized Discounted Cumulative Gain（NDCG）来衡量模型的质量。

在评估示例模型的`RankingMetrics`时，您会注意到DCG和NDCG报告了以下度量值（运行示例时看到的值将类似于这些值）：
* DCG - @1:11.9736, @2:17.5429, @3:21.2532, @4:24.4245, @5:27.0554, @6:29.5571, @7:31.7560, @8:33.7904, @9:35.7949, @10:37.6874

* NDCG: @1:0.4847, @2:0.4820, @3:0.4833, @4:0.4910, @5:0.4977, @6:0.5058, @7:0.5125, @8:0.5182, @9:0.5247, @10:0.5312

NDCG值是最有用的检查，因为这允许我们比较我们的模型在不同数据集的排名能力。NDCG的潜在值从**0.0**到**1.0**不等，其中1.0是一个完美的模型，与理想的排名完全匹配。

考虑到这一点，让我们看看我们的模型的NDCG值。特别是，让我们看看***NDCG@10**的值，即**0.5312**。这是返回前**10**搜索引擎结果的查询的平均NDCG，有助于判断前**10**结果的排名是否正确。为了提高模型的排序能力，需要对特征工程和模型超参数进行实验，并对流水线进行相应的修改。我们将通过修改管道、训练模型和评估度量来继续迭代，直到达到所需的模型质量。

请参阅以下用于训练和评估模型的代码：

```CSharp
// Train the model on the training dataset. To perform training you need to call the Fit() method.
ITransformer model = pipeline.Fit(trainData);

// Load the validation data and use the model to perform predictions on the validation data.
IDataView validationData = mlContext.Data.LoadFromTextFile<SearchResultData>(ValidationDatasetPath, separatorChar: '\t', hasHeader: false);

[...]

// Predict rankings.
IDataView predictions = model.Transform(validationData);

[...]

// Evaluate the metrics for the data using NDCG; by default, metrics for the up to 3 search results in the query are reported (e.g. NDCG@3).
RankingMetrics metrics = mlContext.Ranking.Evaluate(predictions);
`````
### 3. 重新训练并执行模型的最终评估
一旦达到所需的度量，生成的管道将用于组合的**训练+验证**数据集上的训练。然后，我们最后一次使用**测试**数据集评估此模型，以获得模型的最终度量。

请参阅以下代码：

```CSharp
// Train the model on the train + validation dataset.
model = pipeline.Fit(trainValidationData);

// Evaluate the model using the metrics from the testing dataset; you do this only once and these are your final metrics.
IDataView testData = mlContext.Data.LoadFromTextFile<SearchResultData>(TestDatasetPath, separatorChar: '\t', hasHeader: false);

[...]

// Predict rankings.
IDataView predictions = model.Transform(testData);

[...]

// Evaluate the metrics for the data using NDCG; by default, metrics for the up to 3 search results in the query are reported (e.g. NDCG@3).
RankingMetrics metrics = mlContext.Ranking.Evaluate(predictions);

```

### 4. 重新训练并使用模型

最后一步是使用所有数据（**培训+验证+测试**）重新训练模型。

在模型被训练之后，我们可以使用`Predict()` API来预测新的、传入的用户查询的搜索引擎结果排名。

```CSharp
// Retrain the model on all of the data, train + validate + test.
model = pipeline.Fit(allData);

// Save the model
mlContext.Model.Save(model, null, modelPath);

// Load the model to perform predictions with it.
DataViewSchema predictionPipelineSchema;
ITransformer predictionPipeline = mlContext.Model.Load(modelPath, out predictionPipelineSchema);

// Predict rankings.
IDataView predictions = predictionPipeline.Transform(data);

 // In the predictions, get the scores of the search results included in the first query (e.g. group).
 IEnumerable<SearchResultPrediction> searchQueries = mlContext.Data.CreateEnumerable<SearchResultPrediction>(predictions, reuseRowObject: false);
 var firstGroupId = searchQueries.First<SearchResultPrediction>().GroupId;
 IEnumerable<SearchResultPrediction> firstGroupPredictions = searchQueries.Take(100).Where(p => p.GroupId == firstGroupId).OrderByDescending(p => p.Score).ToList();

 // The individual scores themselves are NOT a useful measure of result quality; instead, they are only useful as a relative measure to other scores in the group. 
 // The scores are used to determine the ranking where a higher score indicates a higher ranking versus another candidate result.
 ConsoleHelper.PrintScores(firstGroupPredictions);
`````
