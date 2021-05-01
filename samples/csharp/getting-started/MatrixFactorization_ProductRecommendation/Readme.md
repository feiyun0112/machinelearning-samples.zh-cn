#产品推荐 - 矩阵分解问题示例

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
|Microsoft.ML.Recommender Preview v0.16.0   | 动态 API | 最新版本 | 控制台应用程序 | .txt 文件 | 推荐 | 矩阵分解 | MatrixFactorizationTrainer (One Class)|

在这个示例中，您可以看到如何使用ML.NET来构建产品推荐方案。

本示例中的推荐方式基于共同购买或经常一起购买的产品，这意味着它将根据客户的购买历史向客户推荐一组产品。

![替代文字](https://raw.githubusercontent.com/dotnet/machinelearning-samples/master/samples/csharp/getting-started/MatrixFactorization_ProductRecommendation/ProductRecommender/Data/frequentlyboughttogether.png)

在这个示例中，基于经常一起购买的学习模型来推荐产品。


## 问题
在本教程中，我们将使用亚马逊共同购买产品数据集。

我们将使用One-Class因式分解机来构建我们的产品推荐器，它使用协同过滤方法。

我们介绍的one-class和其他因式分解机的区别在于，在这个数据集中，我们只有购买历史的信息。

我们没有评分或其他详细信息，如产品描述等。

“协同过滤”是在一个基本假设的情况下运作的，即如果某人A在一个问题上与某人B具有相同的意见，则在另一个问题上，相对其他随机选择的人，A更倾向于B的观点。


## 数据集
原始数据来自SNAP:
https://snap.stanford.edu/data/amazon0302.html

[此处](/ProductRecommender/Data/DATASETS-CITATION.txt)为数据集的引文信息

## 算法 - [矩阵分解 (推荐)](https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/tasks#recommendation)

这个推荐任务的算法是矩阵分解，它是一个执行协同过滤的有监督的机器学习任务。

## 解决方案

要解决此问题，您需要在现有训练数据上建立和训练ML模型，评估其有多好（分析获得的指标），最后您可以使用/测试模型来预测给定输入数据变量的需求。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)

### 1. 建立模型

建立模型包括: 

* 从 https://snap.stanford.edu/data/amazon0302.html 下载并复制数据集文件Amazon0302.txt。

* 使用以下内容替换列名：ProductID	ProductID_Copurchased

* 在读取器中，我们已经提供了KeyRange，并且产品ID已经编码，我们需要做的就是使用几个额外的参数调用MatrixFactorizationTrainer。

下面是用于建立模型的代码：
```CSharp
 
    //STEP 1: Create MLContext to be shared across the model creation workflow objects 
    MLContext mlContext = new MLContext();

    //STEP 2: Read the trained data using TextLoader by defining the schema for reading the product co-purchase dataset
    //        Do remember to replace amazon0302.txt with dataset from https://snap.stanford.edu/data/amazon0302.html
    var traindata = mlContext.Data.LoadFromTextFile(path:TrainingDataLocation,
                                                      columns: new[]
                                                                {
                                                                    new TextLoader.Column("Label", DataKind.Single, 0),
                                                                    new TextLoader.Column(name:nameof(ProductEntry.ProductID), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(0) }, keyCount: new KeyCount(262111)), 
                                                                    new TextLoader.Column(name:nameof(ProductEntry.CoPurchaseProductID), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(1) }, keyCount: new KeyCount(262111))
                                                                },
                                                      hasHeader: true,
                                                      separatorChar: '\t');

    //STEP 3: Your data is already encoded so all you need to do is specify options for MatrxiFactorizationTrainer with a few extra hyperparameters
            //        LossFunction, Alpa, Lambda and a few others like K and C as shown below and call the trainer. 
            MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
            options.MatrixColumnIndexColumnName = nameof(ProductEntry.ProductID);
            options.MatrixRowIndexColumnName = nameof(ProductEntry.CoPurchaseProductID);
            options.LabelColumnName= "Label";
            options.LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass;
            options.Alpha = 0.01;
            options.Lambda = 0.025;
            // For better results use the following parameters
            //options.K = 100;
            //options.C = 0.00001;

//Step 4: Call the MatrixFactorization trainer by passing options.
            var est = mlContext.Recommendation().Trainers.MatrixFactorization(options);
```

### 2. 训练模型

一旦定义了评估器，就可以根据可用的训练数据对评估器进行训练。

这将返回一个训练过的模型。

```CSharp

    //STEP 5: Train the model fitting to the DataSet
    //Please add Amazon0302.txt dataset from https://snap.stanford.edu/data/amazon0302.html to Data folder if FileNotFoundException is thrown.
    ITransformer model = est.Fit(traindata);
```

### 3. 使用模型 

我们将通过创建预测引擎/函数来执行此模型的预测。

预测引擎将以下两个类作为输入。

```CSharp
    public class Copurchase_prediction
    {
        public float Score { get; set; }
    }

    public class ProductEntry
    {
            [KeyType(count : 262111)]
            public uint ProductID { get; set; }

            [KeyType(count : 262111)]
            public uint CoPurchaseProductID { get; set; }
    }
```

一旦创建了预测引擎，就可以预测两个产品被共同购买的分数。

```CSharp
    //STEP 6: Create prediction engine and predict the score for Product 63 being co-purchased with Product 3.
    //        The higher the score the higher the probability for this particular productID being co-purchased 
    var predictionengine = mlContext.Model.CreatePredictionEngine<ProductEntry, Copurchase_prediction>(model);
    var prediction = predictionengine.Predict(
                             new ProductEntry()
                             {
                             ProductID = 3,
                             CoPurchaseProductID = 63
                             });
```

#### 矩阵因式分解的得分

矩阵因式分解产生的分数表示为正例的可能性。得分值越大，成为正例的概率越高。但是，分数没有任何概率信息。当你做一个预测时，你必须计算出多个商品的得分，并挑选得分最高的商品。
