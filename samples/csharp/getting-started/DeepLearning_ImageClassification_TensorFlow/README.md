# 图像分类 - 评分示例

## 问题
图像分类是许多业务场景中的常见情况。 对于这些情况，您可以使用预先训练的模型或训练自己的模型来对特定于自定义域的图像进行分类。

## 数据集
 有两个数据源：`tsv`文件和图像文件。[tsv 文件](./ImageClassification/assets/inputs/images/tags.tsv) 包含2列：第一个定义为`ImagePath`，第二个定义为对应于图像的`Label`。正如你所看到的，文件没有标题行，看起来像这样：
```tsv
broccoli.jpg	broccoli
broccoli.png	broccoli
canoe2.jpg	canoe
canoe3.jpg	canoe
canoe4.jpg	canoe
coffeepot.jpg	coffeepot
coffeepot2.jpg	coffeepot
coffeepot3.jpg	coffeepot
coffeepot4.jpg	coffeepot
pizza.jpg	pizza
pizza2.jpg	pizza
pizza3.jpg	pizza
teddy1.jpg	teddy bear
teddy2.jpg	teddy bear
teddy3.jpg	teddy bear
teddy4.jpg	teddy bear
teddy6.jpg	teddy bear
toaster.jpg	toaster
toaster2.png	toaster
toaster3.jpg	toaster
```
训练和测试图像位于assets文件夹中。这些图像属于维基共享资源。
> *[维基共享资源](https://commons.wikimedia.org/w/index.php?title=Main_Page&oldid=313158208), 免费媒体存储库。* 于 10:48, October 17, 2018 检索自:  
> https://commons.wikimedia.org/wiki/Pizza  
> https://commons.wikimedia.org/wiki/Coffee_pot  
> https://commons.wikimedia.org/wiki/Toaster  
> https://commons.wikimedia.org/wiki/Category:Canoes  
> https://commons.wikimedia.org/wiki/Teddy_bear  

## 预训练模型
有多个模型被预先训练用于图像分类。在本例中，我们将使用基于Inception拓扑的模型，并用来自Image.Net的图像进行训练。这个模型可以从 https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip 下载, 也可以在 `/ src / ImageClassification / assets /inputs / inception / tensorflow_inception_graph.pb` 找到。

##  解决方案
控制台应用程序项目`ImageClassification.Score`可用于基于预先训练的Inception-v3 TensorFlow模型对样本图像进行分类。

再次注意，此示例仅使用预先训练的TensorFlow模型和ML.NET API。 因此，它**不会**训练任何ML.NET模型。 目前，在ML.NET中仅支持使用现有的TensorFlow训练模型进行评分/预测。

您需要按照以下步骤执行分类测试：

1) **设置VS默认启动项目：** 将`ImageClassification.Score`设置为Visual Studio中的启动项目。
2)  **运行训练模型控制台应用程序:** 在Visual Studio中按F5。 在执行结束时，输出将类似于此屏幕截图：
![image](./docs/images/train_console.png)


##  代码演练
解决方案中有一个名为`ImageClassification.Score`的项目，它负责以TensorFlow格式加载模型，然后对图像进行分类。

### ML.NET：模型评分
`TextLoader.CreateReader()`用于定义将用于在ML.NET模型中加载图像的文本文件的模式。

```csharp
 var loader = new TextLoader(env,
    new TextLoader.Arguments
    {
        Column = new[] {
            new TextLoader.Column("ImagePath", DataKind.Text, 0)
        }
    });

var data = loader.Read(new MultiFileSource(dataLocation));
```

用于加载图像的图像文件有两列：第一列定义为`ImagePath` ，第二列是与图像对应的`Label`。

需要强调的是，在使用TensorFlow模型进行评分时，这里并没有真正使用标签。该文件仅作为测试预测时的参考，以便您可以将每个样本数据的实际标签与TensorFlow模型提供的预测标签进行比较。这就是为什么当使用上面的'TextLoader'加载文件时，您只需要获取ImagePath或文件的名称，但不需要获取标签。

```csv
broccoli.jpg	broccoli
bucket.png	bucket
canoe.jpg	canoe
snail.jpg	snail
teddy1.jpg	teddy bear
```
正如您所看到的，文件没有标题行。 

第二步是定义估计器流水线。通常，在处理深度神经网络时，必须使图像适应网络期望的格式。这就是为什么图像被调整大小然后被转换的原因（主要是，像素值在所有R、G、B通道上被标准化）。

```csharp
 var pipeline = new ImageLoaderEstimator(env, imagesFolder, ("ImagePath", "ImageReal"))
    .Append(new ImageResizerEstimator(env, "ImageReal", "ImageReal", ImageNetSettings.imageHeight, ImageNetSettings.imageWidth))
    .Append(new ImagePixelExtractorEstimator(env, new[] { new ImagePixelExtractorTransform.ColumnInfo("ImageReal", "input", interleave: ImageNetSettings.channelsLast, offset: ImageNetSettings.mean) }))
    .Append(new TensorFlowEstimator(env, modelLocation, new[] { "input" }, new[] { "softmax2" }));

```
您还需要检查神经网络，并检查输入/输出节点的名称。为了检查模型，可以使用[Netron](https://github.com/lutzroeder/netron)，它会随[Visual Studio Tools for AI](https://visualstudio.microsoft.com/downloads/ai-tools-vs/)一起安装。
这些名称稍后在评估器管道的定义中使用：在初始网络的情况下，输入张量被命名为“input”，输出被命名为“softmax2”。

![inspecting neural network with netron](./docs/images/netron.png)

最后，我们在*拟合*评估器管道之后提取预测函数。 预测函数接收类型为`ImageNetData`的对象（包含2个属性：`ImagePath`和`Label`）作为参数，然后返回类型为`ImagePrediction`的对象。

```
 var modeld = pipeline.Fit(data);
 var predictionFunction = modeld.MakePredictionFunction<ImageNetData, ImageNetPrediction>(env);
```
在获得预测时，我们得到属性`PredictedLabels`中的浮点数数组。数组中的每个位置都被分配给一个标签，例如，如果模型有5个不同的标签，那么数组长度将等于5。数组中的每个位置的值表示标签在该位置上的概率；所有数组值（概率）的总和等于1。然后，您需要选择最大值（概率）并检查指定给该位置的标签。

### 引用
训练和预测图像
> *维基共享资源, 免费媒体存储库。* 于 10:48, October 17, 2018 检索自 https://commons.wikimedia.org/w/index.php?title=Main_Page&oldid=313158208.