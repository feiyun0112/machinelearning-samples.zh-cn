#信用卡欺诈检测（基于异常/异常检测）

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4         | 动态 API | 最新版 | 控制台应用程序 | .csv 文件 | 欺诈检测| 异常检测| Randomized PCA |

在这个介绍性示例中，您将看到如何使用ML.NET来预测信用卡欺诈。在机器学习领域，这种预测被称为异常检测。


##API版本：基于动态和评估器的API

需要注意的是，这个示例使用带有评估器的API。


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


##ML任务-[异常检测](https://en.wikipedia.org/wiki/Anomaly_detection)

异常（或异常值）检测是指通过与大多数数据显著不同而引起怀疑的稀有项目、事件或观察结果的识别。通常，异常项会转化为某种问题，如银行欺诈、结构缺陷、医疗问题或文本中的错误。

如果您想了解如何使用二进制分类检测欺诈，请访问[二进制分类信用卡欺诈检测示例](../BinaryClassification_CreditCardFraudDetection)。

## 解决方案

要解决这个问题，首先需要建立一个机器学习模型。然后，根据现有的训练数据对模型进行训练，评估其准确性有多高，最后您使用该模型（在不同的应用程序中部署内置的模型）来预测样本信用卡交易的欺诈行为。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)


### 1. 建立模型

建立模型包括：

- 为训练和测试准备数据和分割数据。

- 通过指定保存要与数据集映射的数据架构的类型名，使用TextLoader加载数据。

- 创建一个评估器并用`Concatenate()`转换数据，然后按LP Norm规范化。

- 选择训练器/学习算法Randomized PCA对模型进行训练。


初始代码类似于以下代码：

`````csharp

    // Create a common ML.NET context.
    // Seed set to any number so you have a deterministic environment for repeateable results
    MLContext mlContext = new MLContext(seed:1);

[...]
    // Prepare data and create Train/Test split datasets
    PrepDatasets(mlContext, fullDataSetFilePath, trainDataSetFilePath, testDataSetFilePath);

[...]

    //Load the original single dataset
    IDataView originalFullData = mlContext.Data.LoadFromTextFile<TransactionObservation>(fullDataSetFilePath, separatorChar: er: true);
                 
    // Split the data 80:20 into train and test sets, train and evaluate.
    TrainTestData trainTestData = mlContext.Data.TrainTestSplit(originalFullData, testFraction: 0.2, seed: 1);
    IDataView trainData = trainTestData.TrainSet;
    IDataView testData = trainTestData.TestSet;

    
[...]

    // Get all the feature column names (All except the Label and the IdPreservationColumn)
    string[] featureColumnNames = trainDataView.Schema.AsQueryable()
        .Select(column => column.Name)                               // Get alll the column names
        .Where(name => name != nameof(TransactionObservation.Label)) // Do not include the Label column
        .Where(name => name != "IdPreservationColumn")               // Do not include the IdPreservationColumn/StratificationColumn
        .Where(name => name != nameof(TransactionObservation.Time))  // Do not include the Time column. Not needed as feature column
        .ToArray();

    // Create the data process pipeline
    IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms.Concatenate("Features", featureColumnNames)
                                            .Append(mlContext.Transforms.DropColumns(new string[] { nameof(TransactionObservation.Time) }))
                                            .Append(mlContext.Transforms.NormalizeLpNorm(outputColumnName: "NormalizedFeatures",
                                                                                          inputColumnName: "Features"));


    // In Anomaly Detection, the learner assumes all training examples have label 0, as it only learns from normal examples.
    // If any of the training examples has label 1, it is recommended to use a Filter transform to filter them out before training:
    IDataView normalTrainDataView = mlContext.Data.FilterRowsByColumn(trainDataView, columnName: nameof(TransactionObservation.Label), lowerBound: 0, upperBound: 1);

[...]

    var options = new RandomizedPcaTrainer.Options
    {
        FeatureColumnName = "NormalizedFeatures",   // The name of the feature column. The column data must be a known-sized vector of Single.
        ExampleWeightColumnName = null,             // The name of the example weight column (optional). To use the weight column, the column data must be of type Single.
        Rank = 28,                                  // The number of components in the PCA.
        Oversampling = 20,                          // Oversampling parameter for randomized PCA training.
        EnsureZeroMean = true,                      // If enabled, data is centered to be zero mean.
        Seed = 1                                    // The seed for random number generation.
    };


    // Create an anomaly detector. Its underlying algorithm is randomized PCA.
    IEstimator<ITransformer> trainer = mlContext.AnomalyDetection.Trainers.RandomizedPca(options: options);

    EstimatorChain<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer);

`````


### 2. 训练模型

训练模型是在训练数据上运行所选算法以调整模型参数的过程。它是在Estimator对象的`Fit()`方法中实现的。

要执行训练，需要在DataView对象中提供训练数据集（`trainData.csv`）时调用`Fit()`方法。

`````csharp    
    TransformerChain<ITransformer> model = trainingPipeline.Fit(normalTrainDataView);
`````


### 3. 评估模型

我们需要这一步来总结我们的模型有多精确。为此，上一步中的模型将针对训练中未使用的另一个数据集（`testData.csv`）运行。

`Evaluate()`比较测试数据集的预测值，并生成各种指标，例如准确性，您可以对其进探究。 

`````csharp
    EvaluateModel(mlContext, model, testDataView);
`````


### 4. 使用模型

训练完模型后，您可以使用`Predict()`API来预测交易是否存在欺诈。

`````csharp
[...]

    IDataView inputDataForPredictions = mlContext.Data.LoadFromTextFile<TransactionObservation>(_dasetFile, separatorChar: ',', hasHeader: true);

    ITransformer model = mlContext.Model.Load(_modelfile, out var inputSchema);

    var predictionEngine = mlContext.Model.CreatePredictionEngine<TransactionObservation, TransactionFraudPrediction>(model);

[...]

    mlContext.Data.CreateEnumerable<TransactionObservation>(inputDataForPredictions, reuseRowObject: false)
                  .Where(x => x.Label > 0)
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
                  .Where(x => x.Label < 1)
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
