# 欢迎关注我的公众号
<img src="qrcode_for_gh_61af3e28f945_344.jpg">
</img>

> 注意:我们希望听到您对MLOps的反馈。请在[本调查](https://www.research.net/r/mlops-samples)中告诉我们您的想法。

# ML.NET 示例

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) 是一个跨平台的开源机器学习框架，使.NET开发人员使用机器学习变得很容易。

在这个GitHub 存储库中，我们提供了示例，这些示例将帮助您开始使用ML.NET，以及如何将ML.NET加入到现有的和新的.NET应用程序中。

**注意:** 请在[机器学习存储库](https://github.com/dotnet/machinelearning/issues)中打开与[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)框架相关的问题。请仅当您遇到此存储库中的示例问题时，才在存储库中创建该问题。 

存储库中有两种类型的示例/应用程序：

* 入门  ![](./images/app-type-getting-started.png) : 针对每个机器学习任务或领域的ML.NET代码示例，通常作为简单的控制台应用程序实现。 

* 终端应用程序 ![](./images/app-type-e2e.png) : 使用ML.NET进行机器学习的Web，桌面，移动和其他应用程序的实际例子

根据场景和机器学习问题/任务，官方ML.NET示例被分成多个类别，可通过下表访问：

<table align="middle" width=100%>  
  <tr>
    <td align="middle" colspan="3">二元分类</td>
  </tr>
  <tr>
    <td align="middle"><img src="images/sentiment-analysis.png" alt="Binary classification chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>情绪分析<br><a href="samples/csharp/getting-started/BinaryClassification_SentimentAnalysis">C#</a> &nbsp; &nbsp; <a href="samples/fsharp/getting-started/BinaryClassification_SentimentAnalysis">F#</a></b></td>
    <td align="middle"><img src="images/spam-detection.png" alt="Movie Recommender chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>垃圾信息检测<br><a href="samples/csharp/getting-started/BinaryClassification_SpamDetection">C#</a> &nbsp; &nbsp; <a href="samples/fsharp/getting-started/BinaryClassification_SpamDetection">F#</a></b></td>
    <td align="middle"><img src="images/anomaly-detection.png" alt="Power Anomaly detection chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>信用卡欺诈识别<br>(Binary Classification)<br><a href="samples/csharp/getting-started/BinaryClassification_CreditCardFraudDetection">C#</a> &nbsp;&nbsp;&nbsp;<a href="samples/fsharp/getting-started/BinaryClassification_CreditCardFraudDetection">F#</a></b></td>
  </tr> 
  <tr>
    <td align="middle"><img src="images/disease-detection.png" alt="disease detection chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>心脏病预测<br><a href="samples/csharp/getting-started/BinaryClassification_HeartDiseaseDetection">C#</a></td>
    <td></td>
    <td></td>
  </tr> 
  <tr>
    <td align="middle" colspan="3">多类分类</td>
  </tr>
  <tr>
    <td align="middle"><img src="images/issue-labeler.png" alt="Issue Labeler chart"><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>GitHub Issues 分类<br> <a href="samples/csharp/end-to-end-apps/MulticlassClassification-GitHubLabeler">C#</a>&nbsp;&nbsp;<a href="samples/fsharp/end-to-end-apps/MulticlassClassification-GitHubLabeler">F#</a></b></td>
    <td align="middle"><img src="images/flower-classification.png" alt="Movie Recommender chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>鸢尾花分类<br><a href="samples/csharp/getting-started/MulticlassClassification_Iris">C#</a> &nbsp; &nbsp;<a href="samples/fsharp/getting-started/MulticlassClassification_Iris">F#</a></b></td>
    <td align="middle"><img src="images/handwriting-classification.png" alt="Movie Recommender chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>手写数字识别<br><a href="samples/csharp/getting-started/MulticlassClassification_MNIST">C#</a></b></td>
  </tr>
  <tr>
    <td align="middle" colspan="3">建议</td>
  </tr>
  <tr>
    <td align="middle"><img src="images/product-recommendation.png" alt="Product Recommender chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>产品推荐<br><a href="samples/csharp/getting-started/MatrixFactorization_ProductRecommendation">C#</a></h4></td>
    <td align="middle"><img src="images/movie-recommendation.png" alt="Movie Recommender chart" ><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>电影推荐<br>(Matrix Factorization)<b><br><a href="samples/csharp/getting-started/MatrixFactorization_MovieRecommendation">C#</a></b></td>
    <td align="middle"><img src="images/movie-recommendation.png" alt="Movie Recommender chart"><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>电影推荐<br>(Field Aware Factorization Machines)<br><a href="samples/csharp/end-to-end-apps/Recommendation-MovieRecommender">C#</a></b></td>
  </tr>
  <tr>
    <td align="middle" colspan="3">回归</td>
  </tr>
  <tr>
    <td align="middle"><img src="images/price-prediction.png" alt="Price Prediction chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>价格预测<br><a href="samples/csharp/getting-started/Regression_TaxiFarePrediction">C#</a> &nbsp; &nbsp; <a href="samples/fsharp/getting-started/Regression_TaxiFarePrediction">F#</a></b></td>
    <td align="middle"><br><img src="images/sales-forcasting.png" alt="Sales ForeCasting chart"><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>销售预测<br><a href="samples/csharp/end-to-end-apps/Forecasting-Sales">C#</a><br><br></b></td>
    <td align="middle"><img src="images/demand-prediction.png" alt="Demand Prediction chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>需求预测<br><a href="samples/csharp/getting-started/Regression_BikeSharingDemand">C#</a> &nbsp;&nbsp;&nbsp;<a href="samples/fsharp/getting-started/Regression_BikeSharingDemand">F#</a></b></td>
  </tr>
  <tr>
    <td align="middle" colspan="3">时间序列预测</td>
  </tr>
  <tr>
    <td align="middle"><br><img src="images/sales-forcasting.png" alt="Sales ForeCasting chart"><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>销售预测<br><a href="samples/csharp/end-to-end-apps/Forecasting-Sales">C#</a><br><br></b></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td align="middle" colspan="3">异常情况检测</td>
  </tr>
  <tr>
    <td align="middle"><img src="images/spike-detection.png" alt="Spike detection chart"><br><br><b>销售高峰检测<br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon">&nbsp;<a href="samples/csharp/getting-started/AnomalyDetection_Sales">C#</a>&nbsp&nbsp;&nbsp;&nbsp;&nbsp;
      <img src="images/app-type-e2e-black.png" alt="End-to-end app icon">&nbsp;<a href="samples/csharp/end-to-end-apps/AnomalyDetection-Sales">C#</a><b></td>
    <td align="middle"><img src="images/spike-detection.png" alt="Spike detection chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>电力异常检测<br><a href="samples/csharp/getting-started/AnomalyDetection_PowerMeterReadings">C#</a><b></td>
      <td align="middle"><img src="images/anomaly-detection.png" alt="Power Anomaly detection chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>信用卡欺诈检测<br>(Anomaly Detection)<br><a href="samples/csharp/getting-started/AnomalyDetection_CreditCardFraudDetection">C#</a><b></td>
  </tr> 
  <tr>
    <td align="middle" colspan="3">聚类分析</td>
  </tr>
  <tr>
   <td align="middle"><img src="images/customer-segmentation.png" alt="Customer Segmentation chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>客户细分<br><a href="samples/csharp/getting-started/Clustering_CustomerSegmentation">C#</a> &nbsp; &nbsp; <a href="samples/fsharp/getting-started/Clustering_CustomerSegmentation">F#</a></b></td>
    <td align="middle"><img src="images/clustering.png" alt="IRIS Flowers chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>鸢尾花聚类<br><a href="samples/csharp/getting-started/Clustering_Iris">C#</a> &nbsp; &nbsp; <a href="samples/fsharp/getting-started/Clustering_Iris">F#</a></b></td>
    <td></td>
 
  </tr>
  <tr>
    <td align="middle" colspan="3">排名</td>
  </tr>
  <tr>
    <td align="middle"><img src="images/ranking-numbered.png" alt="Ranking chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>排名搜索引擎结果<br><a href="samples/csharp/getting-started/Ranking_Web">C#</a><b></td>
      <td></td>
      <td></td>
  </tr>
  <tr>
    <td align="middle" colspan="3">计算机视觉</td>
  </tr>
  <tr>
  <td align="middle"><img src="images/image-classification.png" alt="Image Classification chart"><br><b>图像分类训练<br>    (High-Level API)<br>
    <img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon">&nbsp;<a href="samples/csharp/getting-started/DeepLearning_ImageClassification_Training">C#</a>&nbsp;<a href="samples/fsharp/getting-started/DeepLearning_ImageClassification_Training">F#</a>&nbsp;&nbsp&nbsp&nbsp&nbsp;&nbsp;
    </td>
    <td align="middle"><img src="images/image-classification.png" alt="Image Classification chart"><br><b>图像分类预测<br>(Pretrained TensorFlow model scoring)<br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon">&nbsp;<a href="samples/csharp/getting-started/DeepLearning_ImageClassification_TensorFlow">C#</a> &nbsp; <a href="samples/fsharp/getting-started/DeepLearning_ImageClassification_TensorFlow">F#</a>&nbsp;&nbsp&nbsp&nbsp&nbsp;&nbsp;
      <img src="images/app-type-e2e-black.png" alt="End-to-end app icon">&nbsp;<a href="samples/csharp/end-to-end-apps/DeepLearning_ImageClassification_TF">C#</a><b></td><b></td>
    <td align="middle"><img src="images/image-classification.png" alt="Image Classification chart"><br><b>图像分类训练<br>    (TensorFlow Featurizer Estimator)<br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon">&nbsp;<a href="samples/csharp/getting-started/DeepLearning_TensorFlowEstimator">C#</a> &nbsp; <a href="samples/fsharp/getting-started/DeepLearning_TensorFlowEstimator">F#</a><b></td>
 
  </tr> 
  <tr>
    <td align="middle"><br><img src="images/object-detection.png" alt="Object Detection chart"><br><b>对象检测<br>    (ONNX model scoring)<br>
    <img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon">&nbsp;<a href="samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx">C#</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="images/app-type-e2e-black.png" alt="End-to-end app icon">&nbsp;<a href="/samples/csharp/end-to-end-apps/ObjectDetection-Onnx">C#</a><b></td>
  </tr> 
</table>

<br>
<br>
<br>

<table >
  <tr>
    <td align="middle" colspan="3">跨领域方案</td>
  </tr>
  <tr>
  <td align="middle"><img src="images/web.png" alt="web image" ><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>Web API上的可扩展模型<br><a href="samples/csharp/end-to-end-apps/ScalableMLModelOnWebAPI-IntegrationPkg">C#</a><b></td>
  <td align="middle"><img src="images/web.png" alt="web image" ><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>Razor Web应用程序上的可扩展模型<br><a href="samples/modelbuilder/BinaryClassification_Sentiment_Razor">C#</a><b></td>
  <td align="middle"><img src="images/azure-functions-20.png" alt="Azure functions logo"><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>Azure Functions上的可扩展模型<br><a href="samples/csharp/end-to-end-apps/ScalableMLModelOnAzureFunction">C#</a><b></td>
</tr>
<tr>
  <td align="middle"><img src="images/smile.png" alt="Database chart"><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>Blazor Web应用程序上的可扩展模型<br><a href="samples/csharp/end-to-end-apps/ScalableSentimentAnalysisBlazorWebApp">C#</a><b></td>
  <td align="middle"><img src="images/large-data-set.png" alt="large file chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>大数据集<br><a href="samples/csharp/getting-started/LargeDatasets">C#</a><b></td>
  <td align="middle"><img src="images/database.png" alt="Database chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>使用DatabaseLoader加载数据<br><a href="samples/csharp/getting-started/DatabaseLoader">C#</a><b></td>
   
  </tr>
  <tr>
    <td align="middle"><img src="images/database.png" alt="Database chart"><br><img src="images/app-type-getting-started-term-cursor.png" alt="Getting started icon"><br><b>使用LoadFromEnumerable加载数据<br><a href="samples/csharp/getting-started/DatabaseIntegration">C#</a><b></td>
  <td align="middle"><img src="images/model-explain-smaller.png" alt="Model explainability chart"><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>模型可解释性<br><a href="samples/csharp/end-to-end-apps/Model-Explainability">C#</a></b></td>
  <td align="middle"><img src="images/extensibility.png" alt="Extensibility icon"><br><img src="images/app-type-e2e-black.png" alt="End-to-end app icon"><br><b>导出到ONNX<br><a href="samples/csharp/getting-started/Regression_ONNXExport">C#</a></b></td>
  </tr>
</table>

# 自动生成ML.NET模型（预览状态）

前面的示例向您展示了如何使用ML.NET API 1.0（发布于2019年5月）。

但是，我们还在努力通过其他技术简化ML.NET的使用，这样您就不需要自己编写代码来训练模型，只需提供数据集即可，ML.NET将为您自动为您自动生成“最佳”模型和运行它的代码。

这些用于自动生成模型的附加技术处于预览状态，目前只支持*二进制分类、多类分类和回归*。在未来的版本中，我们将支持额外的ML任务，如*建议、异常检测、聚类分析等*。

## CLI示例:(预览状态）

ML.NET CLI（命令行界面）是一个可以在任何命令提示符（Windows，Mac或Linux）上运行的工具，用于根据您提供的训练数据集生成高质量的ML.NET模型。 此外，它还生成示例C＃代码以运行/评分该模型以及用于创建/训练它的C#代码，以便您可以研究它使用的算法和设置。

| CLI（命令行界面）示例                  |
|----------------------------------|
| [二元分类示例](/samples/CLI/BinaryClassification_CLI)   |
| [多类分类示例](/samples/CLI/MulticlassClassification_CLI) |
| [回归测试示例](/samples/CLI/Regression_CLI)                |


## 自动化机器学习 API示例:(预览状态）

ML.NET AutoML API基本上是一组打包为NuGet包的库，您可以在.NET代码中使用它们。 AutoML消除了选择不同算法，超参数的任务。 AutoML将智能地生成许多算法和超参数组合，并为您找到高质量的模型。

| 自动化机器学习 API示例                    |
|----------------------------------|
| [二元分类示例](/samples/csharp/getting-started/BinaryClassification_AutoML)   |
| [多类分类示例](/samples/csharp/getting-started/MulticlassClassification_AutoML) |
| [排名示例](/samples/csharp/getting-started/Ranking_AutoML/Ranking) |
| [回归测试示例](/samples/csharp/getting-started/Regression_AutoML)                |
| [高级实验示例](/samples/csharp/getting-started/AdvancedExperiment_AutoML)                |


-------------------------------------------------------
# 其他ML.NET社区示例

除了微软提供的ML.NET示例之外，我们还列出了社区创建的示例，这些示例位于单独的页面中：
[ML.NET 社区示例](https://github.com/dotnet/machinelearning-samples/blob/master/docs/COMMUNITY-SAMPLES.md)

这些社区示例不是由微软维护，而是由其所有者维护。
如果您已经创建了任何很酷的ML.NET示例，请将其信息添加到此[REQUEST issue](https://github.com/dotnet/machinelearning-samples/issues/86) ，我们最终将在上面提到的页面发布其信息。

# 了解更多

教程，机器学习基础知识等详细信息，请参阅[ML.NET指南](https://docs.microsoft.com/en-us/dotnet/machine-learning/) 。

# API参考

请查看[ML.NET API参考](https://docs.microsoft.com/dotnet/api/?view=ml-dotnet)，了解各种可用的 API。

# 贡献

我们欢迎贡献！ 请查看我们的[贡献指南](CONTRIBUTING.md)。

# 社区

请加入我们的Gitter社区 [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

这个项目采用了[贡献者契约](http://contributor-covenant.org/)规定的行为准则，以表明我们社区的预期行为。有关更多信息，请参见[.NET基金会行为准则](https://dotnetfoundation.org/code-of-conduct)。

# 许可证

[ML.NET 示例](https://github.com/dotnet/machinelearning-samples)根据[MIT许可证](https://github.com/dotnet/machinelearning-samples/blob/master/LICENSE)获得许可。
