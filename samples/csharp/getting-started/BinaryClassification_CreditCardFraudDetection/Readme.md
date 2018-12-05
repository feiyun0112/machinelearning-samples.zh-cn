# 基于二元分类和PCA的信用卡欺诈检测

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v0.7           | 动态API | 更新至0.7 | 两个控制台应用程序 | .csv 文件 | 欺诈检测 | 二元分类 | FastTree 二元分类 |

在这个介绍性示例中，您将看到如何使用ML.NET来预测信用卡欺诈。在机器学习领域中，这种类型的预测被称为二元分类。

## API版本：基于动态和评估器的API
请务必注意，此示例使用动态API和评估器。

## 问题
这个问题的核心是预测信用卡交易（及其相关信息/变量）是否是欺诈。
 
交易的输入信息仅包含PCA转换后的数值输入变量。遗憾的是，基于隐私原因，原始特征和附加的背景信息无法得到，但您建立模型的方式不会改变。 

特征V1, V2, ... V28是用PCA获得的主成分，未经PCA转换的特征是“Time”和“Amount”。

“Time”特征包含每个交易和数据集中的第一个交易之间经过的秒数。“Amount”特征是交易金额，该特征可用于依赖于示例的代价敏感学习。特征“Class”是响应变量，如果存在欺诈取值为1，否则为0。

数据集非常不平衡，正类（欺诈）数据占所有交易的0.172％。

使用这些数据集，您可以建立一个模型，当预测该模型时，它将分析交易的输入变量并预测欺诈值为false或true。

## 数据集

训练和测试数据基于公共数据集[dataset available at Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)，其最初来自于Worldline和ULB(Université Libre de Bruxelles)的机器学习小组(ttp://mlg.ulb.ac.be)在研究合作期间收集和分析的数据集。

这些数据集包含2013年9月由欧洲持卡人通过信用卡进行的交易。 这个数据集显示了两天内发生的交易，在284,807笔交易中有492个欺诈。

作者：Andrea Dal Pozzolo、Olivier Caelen、Reid A. Johnson和Gianluca Bontempi。基于欠采样的不平衡分类概率。2015在计算智能和数据挖掘（CIDM）学术研讨会上的发言

有关相关主题的当前和过去项目的更多详细信息，请访问 http://mlg.ulb.ac.be/BruFence 和 http://mlg.ulb.ac.be/ARTML

## 机器学习任务 - [二元分类](https://en.wikipedia.org/wiki/Binary_classification)

二元或二项式分类是根据分类规则将给定集合中的元素分成两组（预测每个元素属于哪个组）的任务。需要决定某项是否具有某种定性属性、某些特定特征的上下文
  
## 解决方案

要解决这个问题，首先需要建立一个机器学习模型。 然后，您可以在现有训练数据上训练模型，评估其准确性有多好，最后使用该模型（在另一个应用程序中部署建立的模型）来预测信用卡交易样本是否存在欺诈。

![Build -> Train -> Evaluate -> Consume](https://raw.githubusercontent.com/dotnet/machinelearning-samples/features/samples-new-api/samples/csharp/getting-started/shared_content/modelpipeline.png)


### 1. 建立模型
建立一个模型包括:

- 定义映射到数据集的数据架构，以便使用DataReader读取

- 拆分训练和测试数据

- 创建一个评估器，并使用ConcatEstimator()转换数据，并通过均值方差进行标准化。 

- 选择一个训练/学习算法（FastTree）来训练模型。


初始代码类似以下内容：

`````csharp

    // Create a common ML.NET context.
    // Seed set to any number so you have a deterministic environment for repeateable results
    MLContext mlContext = new MLContext(seed:1);

[...]
    TextLoader.Column[] columns = new[] {
           // A boolean column depicting the 'label'.
           new TextLoader.Column("Label", DataKind.BL, 30),
           // 29 Features V1..V28 + Amount
           new TextLoader.Column("V1", DataKind.R4, 1 ),
           new TextLoader.Column("V2", DataKind.R4, 2 ),
           new TextLoader.Column("V3", DataKind.R4, 3 ),
           new TextLoader.Column("V4", DataKind.R4, 4 ),
           new TextLoader.Column("V5", DataKind.R4, 5 ),
           new TextLoader.Column("V6", DataKind.R4, 6 ),
           new TextLoader.Column("V7", DataKind.R4, 7 ),
           new TextLoader.Column("V8", DataKind.R4, 8 ),
           new TextLoader.Column("V9", DataKind.R4, 9 ),
           new TextLoader.Column("V10", DataKind.R4, 10 ),
           new TextLoader.Column("V11", DataKind.R4, 11 ),
           new TextLoader.Column("V12", DataKind.R4, 12 ),
           new TextLoader.Column("V13", DataKind.R4, 13 ),
           new TextLoader.Column("V14", DataKind.R4, 14 ),
           new TextLoader.Column("V15", DataKind.R4, 15 ),
           new TextLoader.Column("V16", DataKind.R4, 16 ),
           new TextLoader.Column("V17", DataKind.R4, 17 ),
           new TextLoader.Column("V18", DataKind.R4, 18 ),
           new TextLoader.Column("V19", DataKind.R4, 19 ),
           new TextLoader.Column("V20", DataKind.R4, 20 ),
           new TextLoader.Column("V21", DataKind.R4, 21 ),
           new TextLoader.Column("V22", DataKind.R4, 22 ),
           new TextLoader.Column("V23", DataKind.R4, 23 ),
           new TextLoader.Column("V24", DataKind.R4, 24 ),
           new TextLoader.Column("V25", DataKind.R4, 25 ),
           new TextLoader.Column("V26", DataKind.R4, 26 ),
           new TextLoader.Column("V27", DataKind.R4, 27 ),
           new TextLoader.Column("V28", DataKind.R4, 28 ),
           new TextLoader.Column("Amount", DataKind.R4, 29 )
       };

   TextLoader.Arguments txtLoaderArgs = new TextLoader.Arguments
                                               {
                                                   Column = columns,
                                                   // First line of the file is a header, not a data row.
                                                   HasHeader = true,
                                                   Separator = ","
                                               };


[...]
    var classification = new BinaryClassificationContext(env);

    (trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2);

[...]

    //Get all the column names for the Features (All except the Label and the StratificationColumn)
    var featureColumnNames = _trainData.Schema.GetColumns()
        .Select(tuple => tuple.column.Name) // Get the column names
        .Where(name => name != "Label") // Do not include the Label column
        .Where(name => name != "StratificationColumn") //Do not include the StratificationColumn
        .ToArray();

    var pipeline = _mlContext.Transforms.Concatenate("Features", featureColumnNames)
                    .Append(_mlContext.Transforms.Normalize(inputName: "Features", outputName: "FeaturesNormalizedByMeanVar", mode: NormalizerMode.MeanVariance))                       
                    .Append(_mlContext.BinaryClassification.Trainers.FastTree(labelColumn: "Label", 
                                                                              featureColumn: "Features",
                                                                              numLeaves: 20,
                                                                              numTrees: 100,
                                                                              minDatapointsInLeaves: 10,
                                                                              learningRate: 0.2));

`````

### 2. 训练模型
训练模型是在训练数据（具有已知欺诈值）上运行所选算法以调整模型参数的过程。它是在评估器对象的 `Fit()` 方法中实现。

为了执行训练，您需要在DataView对象中提供了训练数据集（`trainData.csv`）后调用 `Fit()` 方法。

`````csharp    
    var model = pipeline.Fit(_trainData);
`````

### 3. 评估模型
我们需要这一步骤来判定我们的模型对新数据的准确性。 为此，上一步中的模型再次针对另一个未在训练中使用的数据集（`testData.csv`）运行。

`Evaluate()`比较测试数据集的预测值，并生成各种指标，例如准确性，您可以对其进行浏览。 

`````csharp
    var metrics = _context.Evaluate(model.Transform(_testData), "Label");
`````

### 4. 使用模型
训练完模型后，您可以使用`Predict()`API来预测交易是否存在欺诈。

`````csharp
[...]

   ITransformer model;
   using (var file = File.OpenRead(_modelfile))
   {
       model = mlContext.Model.Load(file);
   }

   var predictionFunc = model.MakePredictionFunction<TransactionObservation, TransactionFraudPrediction>(mlContext);

[...]

    dataTest.AsEnumerable<TransactionObservation>(mlContext, reuseRowObject: false)
                        .Where(x => x.Label == true)
                        .Take(numberOfTransactions)
                        .Select(testData => testData)
                        .ToList()
                        .ForEach(testData => 
                                    {
                                        Console.WriteLine($"--- Transaction ---");
                                        testData.PrintToConsole();
                                        predictionFunc.Predict(testData).PrintToConsole();
                                        Console.WriteLine($"-------------------");
                                    });
[...]

    dataTest.AsEnumerable<TransactionObservation>(mlContext, reuseRowObject: false)
                        .Where(x => x.Label == false)
                        .Take(numberOfTransactions)
                        .ToList()
                        .ForEach(testData =>
                                    {
                                        Console.WriteLine($"--- Transaction ---");
                                        testData.PrintToConsole();
                                        predictionFunc.Predict(testData).PrintToConsole();
                                        Console.WriteLine($"-------------------");
                                    });

`````
