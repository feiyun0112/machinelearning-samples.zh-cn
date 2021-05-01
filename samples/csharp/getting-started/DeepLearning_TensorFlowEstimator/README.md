# 图像分类训练（使用TensorFlow Featurizer评估器进行模型合成）


| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4           | 动态 API | 最新版 | 控制台应用程序 | 图像文件 | 图像分类 | Featurization + Classification  | Deep neural network + LbfgsMaximumEntropy |
 
## 问题 
图像分类是一个常见问题，使用机器学习技术已经解决了很长时间。在这个示例中，我们将介绍一种混合了新技术（深度学习）和传统技术（LbfgsMaximumEntropy）的方法。

在这个模型中，我们使用[Inception model](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)作为*特征提取器*（模型已经存储在[assets文件夹](./ImageClassification.Train/assets/inputs/inception/) )。这意味着该模型将通过神经网络处理输入图像，然后将使用分类之前的张量的输出。该张量包含*图像特征*，可用于识别图像。

最后，这些图像特征将输入到LbfgsMaximumEntropy算法/训练器中，该算法将学习如何对不同的图像特征集进行分类。

## 数据集

图像集由训练控制台应用程序“动态”下载。
自动解压缩后的图像集由多个图像文件夹组成。 每个子文件夹对应于您要对未来预测进行分类的图像类（在本例中为花朵类型），如下所示：
```
training-app-folder/assets/inputs/images/flower_photos
    daisy
    dandelion
    roses
    sunflowers
    tulips
```

每个子文件夹的名称很重要，因为它将用作图像分类的标签。

> 此图像集中的所有图像均根据Creative Commons By Attribution许可证获得许可，网址：
> https://creativecommons.org/licenses/by/2.0/
> 下载图像集时，还会下载LICENSE.txt，您可以在其中查看imageset许可证的完整详细信息。

## ML任务 - [图像识别](https://en.wikipedia.org/wiki/Outline_of_object_recognition)
为了解决这个问题，首先我们将构建一个ML模型。然后我们将根据现有数据对模型进行训练，评估它的性能，最后使用该模型对新图像进行分类。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)

### 0. 图像集下载和准备

这是一个样板代码，主要是下载一个.zip文件并解压缩。
一旦图像文件准备好，您就可以使用以下步骤来构建/训练模型。

### 1. 建立模型
构建模型包括以下步骤：
* 从初始数据视图中的文件夹加载图像路径和实际标签
* 将图像加载到内存中，同时根据所使用的TensorFlow预先训练的模型（如InceptionV3）进行转换。（根据所使用的深度神经网络的要求，调整像素值并使其归一化）
* 使用深度神经网络模型进行图像*特征化*
* 使用LbfgsMaximumEntropy进行图像分类

定义数据模式，并在使用TextLoader加载数据时引用该类型。这里的类是ImageData。

```csharp
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }
```

使用工具方法LoadImagesFromDirectory()将所有图像加载到包含整个数据集的初始DataView中：
```csharp
IEnumerable<ImageData> allImages = LoadImagesFromDirectory(folder: fullImagesetFolderPath,
                                                            useFolderNameasLabel: true);

```

随机排列图像，以便在分为两个数据集（训练和测试数据集）之前，可以通过标签类更好地平衡数据集。

```csharp
IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(imageSet);
IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

// Split the data 90:10 into train and test sets, train and evaluate.
TrainTestData trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.10);
IDataView trainDataView = trainTestData.TrainSet;
IDataView testDataView = trainTestData.TestSet;
```

以下步骤定义训练管道。 通常，在处理深度神经网络时，必须使图像适应网络所需的格式。 这就是调整图像大小然后进行转换的原因。

```csharp
// 2. Load images in-memory while applying image transformations 
// Input and output column names have to coincide with the input and output tensor names of the TensorFlow model
// You can check out those tensor names by opening the Tensorflow .pb model with a visual tool like Netron: https://github.com/lutzroeder/netron
// TF .pb model --> input node --> INPUTS --> input --> id: "input" 
// TF .pb model --> Softmax node --> INPUTS --> logits --> id: "softmax2_pre_activation" (Inceptionv1) or "InceptionV3/Predictions/Reshape" (Inception v3)

var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelAsKey, inputColumnName: "Label")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "image_object", imageFolder: imagesFolder, inputColumnName: nameof(DataModels.ImageData.ImagePath)))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image_object_resized", 
                                                            imageWidth: ImageSettingsForTFModel.imageWidth, imageHeight: ImageSettingsForTFModel.imageHeight, 
                                                            inputColumnName: "image_object"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName:"input", inputColumnName:"image_object_resized", 
                                                            interleavePixelColors:ImageSettingsForTFModel.channelsLast, 
                                                            offsetImage:ImageSettingsForTFModel.mean, 
                                                            scaleImage:ImageSettingsForTFModel.scale))  //for Inception v3 needs scaleImage: set to 1/255f. Not needed for InceptionV1. 
                .Append(mlContext.Model.LoadTensorFlowModel(inputTensorFlowModelFilePath).
                        ScoreTensorFlowModel(outputColumnNames: new[] { "InceptionV3/Predictions/Reshape" }, 
                                            inputColumnNames: new[] { "input" }, 
                                            addBatchDimensionInput: false));  // (For Inception v1 --> addBatchDimensionInput: true)  (For Inception v3 --> addBatchDimensionInput: false)
```

最后，添加ML.NET分类培训器（LbfgsMaximumEntropy）以最终确定培训管道：
```csharp
// Set the training algorithm and convert back the key to the categorical values (original labels)                            
var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelAsKey, featureColumnName: "InceptionV3/Predictions/Reshape");  //"softmax2_pre_activation" for Inception v1
var trainingPipeline = dataProcessPipeline.Append(trainer)
                                            .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"));
```

### 2. 训练模型
为了开始训练，请在已建立的管道上执行`Fit`：
```csharp 
  ITransformer model = trainingPipeline.Fit(trainingDataView);
```


### 3. 评估模型
训练结束后，利用训练数据对模型进行评估。`Evaluate`函数需要一个`IDataView`参数，该参数包含使用测试数据集拆分的所有预测，因此我们对模型应用`Transform`，然后取`AsDynamic`值。

```csharp
// Make bulk predictions and calculate quality metrics
ConsoleWriteHeader("Create Predictions and Evaluate the model quality");
IDataView predictionsDataView = model.Transform(testDataView);
           
// Show the performance metrics for the multi-class classification            
var classificationContext = mlContext.MulticlassClassification;
var metrics = classificationContext.Evaluate(predictionsDataView, labelColumnName: LabelAsKey, predictedLabelColumnName: "PredictedLabel");
ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
```

最后，我们保存模型：
```csharp
mlContext.Model.Save(model, predictionsDataView.Schema, outputMlNetModelFilePath);
```

#### 运行应用程序训练模型

您应该按照以下步骤进行操作以便训练模型：
1) 在Visual Studio中将`ImageClassification.Train`设置为启动项目
2) 在Visual Studio中按F5，开始训练流程，所需的时间取决于培训的图像数量。。
3) 训练流程完成后，为了使用新的训练模型更新模型使用程序，必须复制生成的ML.NET模型文件(assets/inputs/imageClassifier.zip)并将其粘贴到模型使用程序(assets/inputs/MLNETModel)和最终用户应用程序（仅运行模型进行预测）。

### 4. 使用模型代码

首先，需要加载在模型训练期间创建的模型
```csharp
ITransformer loadedModel = mlContext.Model.Load(modelLocation,out var modelInputSchema);
```

然后，创建预测引擎，并进行样本预测：
```csharp
var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);

IEnumerable<ImageData> imagesToPredict = LoadImagesFromDirectory(imagesFolder, true);

//Predict the first image in the folder
//
ImageData imageToPredict = new ImageData
{
    ImagePath = imagesToPredict.First().ImagePath
};

var prediction = predictionEngine.Predict(imageToPredict);

Console.WriteLine("");
Console.WriteLine($"ImageFile : [{Path.GetFileName(imageToPredict.ImagePath)}], " +
                    $"Scores : [{string.Join(",", prediction.Score)}], " +
                    $"Predicted Label : {prediction.PredictedLabelValue}");

```
预测引擎接收`ImageData`类型的对象作为参数（包含两个属性：`ImagePath`和`Label`）。然后返回`ImagePrediction`类型的对象，该对象包含`PredictedLabel`和 `Score`（*概率*值介于0和1之间）属性。

#### 模型测试：进行分类
1) 复制由训练数据集生成的模型（[ImageClassification.Train](./ImageClassification.Train/)/[assets](./ImageClassification.Train/assets/)/[outputs](./ImageClassification.Train/assets/outputs/)/[imageClassifier.zip](./ImageClassification.Train/assets/outputs/imageClassifier.zip)）到预测项目（[ImageClassification.Predict](./ImageClassification.Predict/)/[assets](./ImageClassification.Predict/assets/)/[inputs](./ImageClassification.Predict/assets/inputs/)/[MLNETModel](./ImageClassification.Predict/assets/inputs/MLNETModel)/[imageClassifier.zip](./ImageClassification.Predict/assets/inputs/imageClassifier.zip)）。
2) 设置VS默认启动项目：在Visual Studio中将`ImageClassification.Predict`设置为启动项目。
3) 在Visual Studio中按F5。 几秒钟后，该过程将完成并显示预测。
