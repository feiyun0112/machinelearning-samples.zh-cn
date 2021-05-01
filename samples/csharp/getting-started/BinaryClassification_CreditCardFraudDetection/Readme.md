# 信用卡欺诈识别（二元分类）

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4           | 动态 API | 最新版 | 两个控制台应用程序 | .csv 文件 | 欺诈识别 | 二元分类 | FastTree 二元分类 |

在这个介绍性示例中，您将看到如何使用ML.NET来预测信用卡欺诈。在机器学习领域中，这种类型的预测被称为二元分类。

## 问题
这个问题的核心是预测信用卡交易（及其相关信息/变量）是否是欺诈。
 
交易的输入信息仅包含PCA（主成分分析）转换后的数值输入变量。遗憾的是，基于隐私原因，原始特征和附加的背景信息无法得到，但您建立模型的方式不会改变。 

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

如果您想了解如何使用异常检测来检测欺诈，请访问[异常检测信用卡欺诈检测示例](../AnomalyDetection_CreditCardFraudDetection).。

## 解决方案

要解决这个问题，首先需要建立一个机器学习模型。 然后，您可以在现有训练数据上训练模型，评估其准确性有多好，最后使用该模型（在另一个应用程序中部署建立的模型）来预测信用卡交易样本是否存在欺诈。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)


### 1. 建立模型
建立一个模型包括:

- 准备数据并拆分为训练和测试数据

- 使用TextLoader加载数据，并指定与数据集映射的数据架构的类型名称。

- 创建一个估算器，并用Concatenate()转换数据，然后通过均值方差进行标准化。 

- 选择一个训练/学习算法（FastTree）来训练模型。


初始代码类似以下内容：

`````csharp

    // Create a common ML.NET context.
    // Seed set to any number so you have a deterministic environment for repeateable results
    MLContext mlContext = new MLContext(seed:1);

[...]

// Prepare data and create Train/Test split datasets
    PrepDatasets(mlContext, fullDataSetFilePath, trainDataSetFilePath, testDataSetFilePath);

[...]

// Load Datasets
IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TransactionObservation>(trainDataSetFilePath, separatorChar: ',', hasHeader: true);
IDataView testDataView = mlContext.Data.LoadFromTextFile<TransactionObservation>(testDataSetFilePath, separatorChar: ',', hasHeader: true);

    
[...]

   //Get all the feature column names (All except the Label and the IdPreservationColumn)
    string[] featureColumnNames = trainDataView.Schema.AsQueryable()
        .Select(column => column.Name)                               // Get alll the column names
        .Where(name => name != nameof(TransactionObservation.Label)) // Do not include the Label column
        .Where(name => name != "IdPreservationColumn")               // Do not include the IdPreservationColumn/StratificationColumn
        .Where(name => name != "Time")                               // Do not include the Time column. Not needed as feature column
        .ToArray();

    // Create the data process pipeline
    IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms.Concatenate("Features", featureColumnNames)
                                            .Append(mlContext.Transforms.DropColumns(new string[] { "Time" }))
                                            .Append(mlContext.Transforms.NormalizeMeanVariance(inputColumnName: "Features",
                                                                                 outputColumnName: "FeaturesNormalizedByMeanVar"));

    // Set the training algorithm
    IEstimator<ITransformer> trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: nameof(TransactionObservation.Label),
                                            featureColumnName: "FeaturesNormalizedByMeanVar",
                                            numberOfLeaves: 20,
                                            numberOfTrees: 100,
                                            minimumExampleCountPerLeaf: 10,
                                            learningRate: 0.2);

`````

### 2. 训练模型
训练模型是在训练数据（具有已知欺诈值）上运行所选算法以调整模型参数的过程。它是在估算器对象的 `Fit()` 方法中实现。

为了执行训练，您需要在DataView对象中提供了训练数据集后调用 `Fit()` 方法。

`````csharp    
    ITransformer model = pipeline.Fit(trainingDataView);
`````

### 3. 评估模型
我们需要这一步骤来判定我们的模型对新数据的准确性。 为此，上一步中的模型再次针对另一个未在训练中使用的数据集运行。

`Evaluate()`比较测试数据集的预测值，并生成各种指标，例如准确性，您可以对其进探究。 

`````csharp
    EvaluateModel(mlContext, model, testDataView, trainerName);
`````

### 4. 使用模型
训练完模型后，您可以使用`Predict()`API来预测交易是否存在欺诈。

`````csharp
[...]

   ITransformer model;
   DataViewSchema inputSchema;
   using (var file = File.OpenRead(_modelfile))
   {
       model = mlContext.Model.Load(file, out inputSchema);
   }

   var predictionEngine = mlContext.Model.CreatePredictionEngine<TransactionObservation, TransactionFraudPrediction>(model);

[...]

    mlContext.Data.CreateEnumerable<TransactionObservation>(inputDataForPredictions, reuseRowObject: false)
                        .Where(x => x.Label == true)
                        .Take(numberOfPredictions)
                        .Select(testData => testData)
                        .ToList()
                        .ForEach(testData => 
                                    {
                                        Console.WriteLine($"--- Transaction ---");
                                        testData.PrintToConsole();
                                        predictionEngine.Predict(testData).PrintToConsole();
                                        Console.WriteLine($"-------------------");
                                    });
[...]

    mlContext.Data.CreateEnumerable<TransactionObservation>(inputDataForPredictions, reuseRowObject: false)
                        .Where(x => x.Label == false)
                        .Take(numberOfPredictions)
                        .ToList()
                        .ForEach(testData =>
                                    {
                                        Console.WriteLine($"--- Transaction ---");
                                        testData.PrintToConsole();
                                        predictionEngine.Predict(testData).PrintToConsole();
                                        Console.WriteLine($"-------------------");
                                    });

`````
