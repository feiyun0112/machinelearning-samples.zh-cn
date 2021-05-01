# GitHub Issues Labeler

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4            | 动态 API | 最新版 | 控制台应用程序 | .csv 文件 和 GitHub 问题 | 问题分类 | 多类分类 | SDCA 多类分类器, AveragedPerceptronTrainer |


这是一个简单的原型应用程序，演示如何使用[ML.NET](https://www.nuget.org/packages/Microsoft.ML/) APIs。主要的重点是创建、训练和使用在 Predictor.cs 类中实现的ML（机器学习）模型。

## 概述
GitHubLabeler 是一个.NET Core控制台应用程序， 它的功能如下:
* 在被标记的GitHub问题上训练ML模型，以教导模型如何为新问题分配标签。 （例如，您可以使用`corefx-issues-train.tsv`文件，该文件包含来自包含来自公共的[corefx](https://github.com/dotnet/corefx)存储库的问题）
* 标记新问题。 应用程序将从`appsettings.json`文件中指定的GitHub存储库中获取所有未标记的未解决问题，并使用在上面步骤中创建的受过训练的ML模型对其进行标记

这个ML模型使用[ML.NET](https://www.nuget.org/packages/Microsoft.ML/)的多类分类算法（`SdcaMultiClassTrainer`）。

## 输入您的GitHub配置数据
1. 在`appsettings.json`文件中**提供您的GitHub数据**:

    为了允许应用程序在GitHub存储库中标记问题，您需要向 appsettings.json 文件中提供以下数据。
    ```csharp
        {
          "GitHubToken": "您的GitHub Token",
          "GitHubRepoOwner": "您的存储库所有者或组织",
          "GitHubRepoName": "您的存储库唯一名"
        }
    ```
    您的用户帐户（`GitHubToken`）应具有对存储库（`GitHubRepoName`）的写入权限。

    点击这里查看[如何创建GitHub Token](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/)。

    `GitHubRepoOwner`可以是GitHub用户ID（比如“MyUser”），也可以是GitHub组织（比如“dotnet”）。

2. **提供训练文件**

    a.  您可以使用现有的`corefx_issues.tsv`数据文件来体验该程序。 在这种情况下，将从[corefx](https://github.com/dotnet/corefx)存储库的标签中选择预测的标签。 无需更改。
    
    b. 要使用GitHub存储库中的标签，您需要在数据上训练模型。为此，请从您的存储库中导出GitHub问题到`.tsv`文件，文件包含以下几列：
    * ID - 问题 ID
    * Area - 问题的标签（以这种方式命名以避免与ML.NET中的Label概念混淆）
    * Title - 问题的标题
    * Description - 问题的描述
    
    将文件添加到`Data`文件夹下。更新`DataSetLocation`部分以匹配您的文件名：    
```csharp
private static string DataSetLocation = $"{BaseDatasetsLocation}/corefx-issues-train.tsv";
```

## 训练 
训练是通过已知示例（在本例中，是包含标签的问题）运行ML模型并教授它如何标记新问题的过程。在这个示例中，它是通过在控制台应用程序调用下列方法来完成：
```csharp
BuildAndTrainModel(DataSetLocation, ModelFilePathName);
```
训练完成后，模型将保存为`MLModels\GitHubLabelerModel.zip`。

## 标记
当模型被训练后，它可以用于预测新问题的标签。

对于没有连接到真正的GitHub存储库的单个测试/演示，请在控制台应用程序中调用下列方法：
```csharp
TestSingleLabelPrediction(ModelFilePathName);
```

要访问GitHub存储库的真实问题，请在控制台应用程序中调用另一个方法：
```csharp
await PredictLabelsAndUpdateGitHub(ModelFilePathName);
```

为了便于在从GitHub存储库中读取问题时进行测试，它只会加载过去10分钟中创建的并且需要标记的未标记问题。 但是您可以修改这个配置：
```csharp
Since = DateTime.Now.AddMinutes(-10)
```
您可以修改这些设置。 在预测标签后，程序会使用预测的标签更新GitHub存储库中的问题。
