# ML.NET 示例
[![](https://dotnet.visualstudio.com/_apis/public/build/definitions/9ee6d478-d288-47f7-aacc-f6e6d082ae6d/22/badge)](https://dotnet.visualstudio.com/public/_build/index?definitionId=22 )
[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) 是一个跨平台的开源机器学习框架，使.NET开发人员使用机器学习变得很容易。在这个GitHub 存储库中，我们提供了示例，这些示例将帮助您开始使用ML.NET，以及如何将ML.NET加入到现有的和新的.NET应用程序中。

存储库中有两种类型的示例/应用程序：

* ![](https://github.com/dotnet/machinelearning-samples/blob/features/samples-new-api/images/app-type-getting-started.png)  入门 - 针对每个机器学习任务或领域的ML.NET代码示例，通常作为简单的控制台应用程序实现。 

* ![](https://github.com/dotnet/machinelearning-samples/blob/features/samples-new-api/images/app-type-e2e.png)  终端应用程序 - 使用ML.NET进行机器学习的Web，桌面，移动和其他应用程序的实际例子

根据场景和机器学习问题/任务，官方ML.NET示例被分成多个类别，可通过下表访问：

<table>
 <tr>
   <td width="25%">
      <h3><b>机器学习任务</b></h3>
  </td>
  <td>
      <h3 width="35%"><b>说明</b></h3>
  </td>
  <td>
      <h3><b>场景</b></h3>
  </td>
 </tr>
 <tr>
   <td width="25%">
      <h3>二元分类</h3>
      <img src="images/Binary Classification.png" alt="二元分类 图表" width="120" height="120"  align="middle">
  </td>
  <td width="35%">
  将给定集合中的元素分类为两组，预测每个元素属于哪个组。
  </td>
    <td>
      <h4>情绪分析 &nbsp;&nbsp;&nbsp;
      <a href="samples/csharp/getting-started/BinaryClassification_SentimentAnalysis">C#（已翻译）</a> &nbsp; &nbsp; <a href="samples/fsharp/getting-started/BinaryClassification_SentimentAnalysis">F#</a>&nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
      <h4>垃圾信息检测 &nbsp;&nbsp;&nbsp;
      <a href="samples/csharp/getting-started/BinaryClassification_SpamDetection">C#（已翻译）</a>&nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
      <h4>欺诈识别 &nbsp;&nbsp;&nbsp;<a href="samples/csharp/getting-started/BinaryClassification_CreditCardFraudDetection">C#（已翻译）</a> &nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
  </td>
 </tr>
 <tr>
   <td width="25%">
      <h3>多类分类</h3>
      <img src="images/Multiple Classification.png" alt="多类分类" width="120" height="120"  align="middle">
  </td>
  <td width="35%">
  将实例分类为三个或更多个类中的一个，预测每个类属于哪个组。
  </td>
  <td>
      <h4>问题分类 &nbsp;&nbsp;&nbsp;
      <a href="samples/csharp/end-to-end-apps/MulticlassClassification-GitHubLabeler">C#</a> &nbsp;&nbsp;&nbsp;<img src="images/app-type-e2e.png" alt="终端应用程序图标"></h4>
      <h4>鸢尾花分类 &nbsp;&nbsp;&nbsp;<a href="samples/csharp/getting-started/MulticlassClassification_Iris">C#</a> &nbsp; &nbsp;<a href="samples/fsharp/getting-started/MulticlassClassification_Iris">F#</a> &nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
  </td>
 </tr>
 <tr>
   <td width="25%">
      <h3>回归</h3>
      <img src="images/Regression.png" alt="回归图标" width="120" height="120"  align="middle">
  </td>
  <td width="35%">
  用给定的输入变量数据预测一个数值。广泛用于预报和预测。
  </td>
  <td>
      <h4>价格预测 &nbsp;&nbsp;&nbsp;
      <a href="samples/csharp/getting-started/Regression_TaxiFarePrediction">C#</a> &nbsp; &nbsp; <a href="samples/fsharp/getting-started/Regression_TaxiFarePrediction">F#</a>&nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
      <h4>销售预测 &nbsp;&nbsp;&nbsp;
      <a href="samples/csharp/end-to-end-apps/Regression-SalesForecast">C#</a>  &nbsp;&nbsp;&nbsp;<img src="images/app-type-e2e.png" alt="终端应用程序图标"></h4>
      <h4>需求预测 &nbsp;&nbsp;&nbsp;
      <a href="samples/csharp/getting-started/Regression_BikeSharingDemand">C#</a> &nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
  </td>
 </tr>
 <tr>
   <td width="25%">
      <h3>建议</h3>
      <img src="images/Recommendation.png" alt="建议图标" width="120" height="120"  align="middle">
  </td>
  <td width="35%">
  推荐系统通常使用基于内容和基于协同过滤的算法。 协同过滤算法基于用户过去的行为/喜好/评分来预测他可能喜欢的项目/产品。
  </td>
  <td>
      <h4>电影推荐 &nbsp;&nbsp;&nbsp;
        <a href="samples/csharp/getting-started/MatrixFactorization_MovieRecommendation">C#</a> &nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标">
        <a href="samples/csharp/end-to-end-apps/Recommendation-MovieRecommender">C#</a> &nbsp;&nbsp;&nbsp;<img src="images/app-type-e2e.png" alt="终端应用程序图标"> </h4>
       <h4>产品推荐  即将推出 &nbsp;&nbsp;&nbsp;<img src="images/app-type-e2e.png" alt="终端应用程序图标"></h4>
  </td>
 </tr>
  <tr>
   <td width="25%">
      <h3>聚类</h3>
      <img src="images/Clustering.png" alt="聚类绘图" width="120" height="120"  align="middle">
  </td>
  <td width="35%">
  以一种方式对一组对象进行分组的机器学习任务，使得同一组中的对象（称为群集）彼此更相似，而不是与其他组中的对象相似。 这是一项探索性任务。 它不会把项目分类到特定的标签上。
  </td>
  <td>
      <h4>客户细分 &nbsp;&nbsp;&nbsp;
      <a href="samples/csharp/getting-started/Clustering_CustomerSegmentation">C#</a> &nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
      <h4>鸢尾花聚类 &nbsp;&nbsp;&nbsp;
      <a href="samples/csharp/getting-started/Clustering_Iris">C#</a> &nbsp; &nbsp; <a href="samples/fsharp/getting-started/Clustering_Iris">F#</a>&nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
  </td>
 </tr>
  <tr>
   <td width="25%">
      <h3>异常检测</h3>
      <img src="images/Anomaly Detection.png" alt="异常检测图表" width="120" height="120"  align="middle">
  </td>
  <td width="35%">
任务的目标是识别稀有项目，事件或观测数据，它们与大多数数据显著不同，从而引起怀疑。通常是诸如银行欺诈，结构缺陷或医疗问题等
  </td>
  <td>
      <h4>即将推出</h4>
  </td>
 </tr>
  <tr>
   <td width="25%">
      <h3>排名</h3>
      <img src="images/Ranking.png" alt="排名标志" width="120" height="120"  align="middle">
  </td>
  <td width="35%">
  为信息检索系统构建排名模型，以便根据用户的输入变量，如喜欢/不喜欢、语境、兴趣等对项目进行排序/排名。
  </td>
  <td>
      <h4>即将推出</h4>
  </td>
 </tr>
  <tr>
   <td width="25%">
      <h3>深度学习</h3>
      <img src="images/Deep Learning.png" alt="深度学习标志" width="120" height="120"  align="middle">
  </td>
  <td width="35%">
  深度学习是机器学习的一个子集。深层学习架构，如深度神经网络，通常应用于诸如计算机视觉（目标检测、图像分类、风格转移）、语音识别、自然语言处理和音频识别等领域。
  </td>
  <td>
      <h4>集成TensorFlow &nbsp;&nbsp;&nbsp;<a href="samples/csharp/getting-started/DeepLearning_ImageClassification_TensorFlow">C#</a> &nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
      <h4>目标检测 即将推出 &nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
      <h4>风格转移  即将推出 &nbsp;&nbsp;&nbsp;<img src="images/app-type-e2e.png" alt="终端应用程序图标"></h4>
      <h4>集成ONNX - 即将推出 &nbsp;&nbsp;&nbsp;<img src="images/app-type-getting-started.png" alt="入门图标"></h4>
  </td>
 </tr>
 </table>

**配置NuGet源:** 通常，您只需要使用常规的NuGet源 https://api.nuget.org/v3/index.json ， 然而，在发布次要版本（例如0.8、0.9等）之前的几天内，我们将使用MyGet中可用的预览版NuGet包（例如0.8.0-preview-27128-1），这在常规NuGet源中不可用。

如果是这种情况，请在Visual Studio中配置MyGet源：

https://dotnet.myget.org/F/dotnet-core/api/v3/index.json

-------------------------------------------------------

## 其他社区示例

除了微软提供的ML.NET示例之外，我们还列出了社区创建的示例，这些示例位于单独的页面中：
[ML.NET 社区示例](https://github.com/dotnet/machinelearning-samples/blob/master/docs/COMMUNITY-SAMPLES.md)

这些社区示例不是由微软维护，而是由其所有者维护。
如果您已经创建了任何很酷的ML.NET示例，请将其信息添加到此[REQUEST issue](https://github.com/dotnet/machinelearning-samples/issues/86) ，我们最终将在上面提到的页面发布其信息。

## 了解更多

教程，机器学习基础知识等详细信息，请参阅[ML.NET指南](https://docs.microsoft.com/en-us/dotnet/machine-learning/) 。

## API参考

请查看[ML.NET API参考](https://docs.microsoft.com/dotnet/api/?view=ml-dotnet)，了解各种可用的 API。

## 贡献

我们欢迎贡献！ 请查看我们的[贡献指南](CONTRIBUTING.md)。

## 社区

请加入我们的Gitter社区 [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

这个项目采用了[贡献者盟约](http://contributor-covenant.org/)规定的行为准则，以表明我们社区的预期行为。有关更多信息，请参见[.NET基金会行为准则](https://dotnetfoundation.org/code-of-conduct)。

## 许可证

[ML.NET 示例](https://github.com/dotnet/machinelearning-samples)根据[MIT许可证](https://github.com/dotnet/machinelearning-samples/blob/master/LICENSE)获得许可。
