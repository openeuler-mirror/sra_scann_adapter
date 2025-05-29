# sra_scann_adapter

### 1 介绍

Adapter for Kunpeng ScaNN Library

### 2 使用说明

#### 2.1 安装SRA_Recall

请参照  [《鲲鹏召回算法库》](https://www.hikunpeng.com/document/detail/zh/SRA/accelFeatures/recall/kunpengsra_recall_16_0030.html)开发指南>安装配置环境>SRA_Recall使用说明>安装SRA_Recall 章节进行操作。

#### 2.2 生成完整的ScaNN

KScaNN依赖基于鲲鹏优化的开源ScaNN以提供完整的功能。因此安装SRA_Recall后需自行获取基于鲲鹏优化的开源ScaNN代码，以编译出完整的ScaNN的Python安装包及动态库文件。

请参照 [《鲲鹏召回算法库》](https://www.hikunpeng.com/document/detail/zh/SRA/accelFeatures/recall/kunpengsra_recall_16_0030.html)开发指南>安装配置环境>SRA_Recall使用说明>生成完整的ScaNN  章节进行操作。

### 3 测试方法指导

#### 3.1 基础数据集介绍

| 数据集名称         | 向量维度 | 训练集大小   | 测试集大小 | 距离度量      | 数据集描述    |
| ------------- | ---- | ------- | ----- | --------- |:-------- |
| GloVe         | 100  | 1183514 | 10000 | Angular   | 英文单词向量   |
| DEEP1B        | 96   | 9990000 | 10000 | Angular   | 图像向量     |
| GIST          | 960  | 1000000 | 1000  | Euclidean | 图像GIST特征 |
| SIFT          | 128  | 1000000 | 10000 | Euclidean | 图像GIST特征 |
| Fashion-MNIST | 784  | 60000   | 10000 | Euclidean | 灰度图像向量   |

#### 3.2 操作步骤

Python接口测试：

请参照  [《鲲鹏召回算法库》](https://www.hikunpeng.com/document/detail/zh/SRA/accelFeatures/recall/kunpengsra_recall_16_0030.html)开发指南>KScaNN接口说明>Python接口>使用示例  章节进行操作。

C++接口测试：

请参照  [《鲲鹏召回算法库》](https://www.hikunpeng.com/document/detail/zh/SRA/accelFeatures/recall/kunpengsra_recall_16_0030.html)开发指南>KScaNN接口说明>C++接口>使用示例  章节进行操作。
