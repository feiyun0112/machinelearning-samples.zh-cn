# 心脏病预测 

| ML.NET 版本 | API 类型          | 状态                        | 应用程序类型    | 数据类型 | 场景            | 机器学习任务                   | 算法                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.4           | 动态 API | 最新版 | 控制台应用程序 | .txt 文件 | 心脏病预测 | 二元分类 | FastTree |

在这个介绍性示例中，您将看到如何使用[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet)预测心脏病。在机器学习领域中，这种类型的预测被称为**二元分类**。

## 数据集
数据集使用的是: [UCI Heart disease] (https://archive.ics.uci.edu/ml/datasets/heart+Disease)
此数据库包含76个属性，但所有已发布的实验都只使用了其中14个属性的子集。

该数据集的引用文献可在此处获取[DataSets-Citation](./HeartDiseaseDetection/Data/DATASETS-CITATION.txt)

## 问题
该问题集中在基于14个属性预测是否有心脏病。 为了解决这个问题，我们将构建一个ML模型，它将以14列作为输入，其中13列是特征列（也称为自变量），再加上您要预测的'Label'列，本示例中为'num' ：

属性信息:

* (age) - 年龄
* (sex) - (1 = 男性; 0 = 女性) 
* (cp) - 胸痛类型 - 值1：典型心绞痛 - 值2：非典型心绞痛 - 值3：非心绞痛 - 值4：无症状
* (trestbps) - 静息血压（入院时单位：mm Hg）
* (chol) - 血清胆固醇（mg/dl） 
* (fbs) -（空腹血糖>120 mg/dl）（1=是；0=否）
* (restecg) - 心电图检查结果——值0：正常——值1：有ST-T波异常（T波倒置和/或ST抬高或降低>0.05 mV）——值2：根据ESTES标准显示可能或确定的左室肥大。
* (thalach) - 最大心率
* (exang) - 运动性心绞痛（1=是；0=否）
* (oldpeak) - 运动引起ST段压低
* (slope) - 峰值运动ST段的斜率——值1：上坡——值2：平坡——值3：下坡 
* (ca) - 用荧光染色的主要血管数（0-3）
* (thal) - 3 =正常; 6 =固定缺陷; 7 =可逆缺陷
* (num) -（预测属性）心脏病诊断（血管造影疾病状态）--值0：<50%直径变窄--值1：>50%直径变窄

并预测患者心脏病的存在，整数值从0到4：
克利夫兰数据库（本例中使用的数据集）的实验集中于简单地尝试区分存在（值1）和不存在（值0）。 


## 机器学习任务 - 二元分类
**二元分类**一般用于将项目分类为两个类中的一个的问题（将项目分类为两个以上的类称为**多类分类**）。

* 预测保险索赔是否有效。
* 预测飞机是否会延误或将准时到达。
* 预测face ID（照片）是否属于设备的所有者。

所有这些示例的共同特征是我们想要预测的参数只能采用两个值中的一个。 换句话说，该值由 `boolean` 类型表示。

## 解决方案
要解决这个问题，首先我们将建立一个机器学习模型。然后，我们将在现有数据上训练模型，评估其有多好，最后我们将使用该模型来预测心脏病是否存在。

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)

### 1. 建立模型

建立模型包括: 

* 定义要使用TextLoader加载（`HeartTraining.tsv` 和 `HeartTest.csv`）到数据集的数据模式。

* 通过将特征连接到单个“features”列来创建估算器

* 选择训练器/学习算法(比如`FastTree`)来训练模型。

初始代码类似以下内容：

```CSharp
// STEP 1: Common data loading configuration
var trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataPath, hasHeader: true, separatorChar: ';');
var testDataView = mlContext.Data.LoadFromTextFile<HeartData>(TestDataPath, hasHeader: true, separatorChar: ';');

// STEP 2: Concatenate the features and set the training algorithm
var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));                         

```

### 2. 训练模型
训练模型是在训练数据上运行所选算法以调整模型参数的过程。它是在估算器对象的 `Fit()` 方法中实现。

为了执行训练，您需要在为DataView对象提供了训练数据集后调用 `Fit()` 方法。

```CSharp
ITransformer trainedModel = pipeline.Fit(trainingDataView);
```

请注意，ML.NET使用延迟加载方式处理数据，所以在实际调用.Fit()方法之前，没有任何数据真正加载到内存中。

### 3. 评估模型

我们需要这一步骤来判定我们的模型对新数据的准确性。 为此，上一步中的模型再次针对测试数据集(`HeartTest.csv`)运行。 此数据集包含了已知的标签。

`Evaluate()`将测试数据集与预测值进行比较，并生成各种指标，例如准确性，您可以对其进行探究。 

```CSharp
var predictions = trainedModel.Transform(testDataView);
var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
```

### 4. 使用模型

训练完模型后，您可以使用`Predict()`API来预测心脏病是否出现在心脏数据集列表中。

```CSharp
// Create prediction engine related to the loaded trained model
var predictionEngine = mlContext.Model.CreatePredictionEngine<HeartData, HeartPrediction>(trainedModel);                   

foreach (var heartData in HeartSampleData.heartDataList)
            {
                var prediction = predictionEngine.Predict(heartData);

                Console.WriteLine($"=============== Single Prediction  ===============");
                Console.WriteLine($"Age: {heartData.Age} ");
                Console.WriteLine($"Sex: {heartData.Sex} ");
                Console.WriteLine($"Cp: {heartData.Cp} ");
                Console.WriteLine($"TrestBps: {heartData.TrestBps} ");
                Console.WriteLine($"Chol: {heartData.Chol} ");
                Console.WriteLine($"Fbs: {heartData.Fbs} ");
                Console.WriteLine($"RestEcg: {heartData.RestEcg} ");
                Console.WriteLine($"Thalac: {heartData.Thalac} ");
                Console.WriteLine($"Exang: {heartData.Exang} ");
                Console.WriteLine($"OldPeak: {heartData.OldPeak} ");
                Console.WriteLine($"Slope: {heartData.Slope} ");
                Console.WriteLine($"Ca: {heartData.Ca} ");
                Console.WriteLine($"Thal: {heartData.Thal} ");
                Console.WriteLine($"Prediction Value: {prediction.Prediction} ");
                Console.WriteLine($"Prediction: {(prediction.Prediction ? "A disease could be present" : "Not present disease" )} ");
                Console.WriteLine($"Probability: {prediction.Probability} ");
                Console.WriteLine($"==================================================");
                Console.WriteLine("");
                Console.WriteLine("");
            }

```


