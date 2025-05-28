# sra_scann_adapter

### 1 Introduction

Adapter for Kunpeng ScaNN Library  

### 2 Usage Instructions

#### 2.1 Installing SRA_Recall

Please follow the **[Kunpeng Recall Algorithm Library](https://www.hikunpeng.com/document/detail/en/SRA/accelFeatures/recall/kunpengsra_recall_16_0030.html) Developer Guide > Installing and Configuring the Environment > Using SRA_Recall > Installing SRA_Recall** section for installation.  

#### 2.2 Generating a Complete ScaNN Library

KScaNN relies on the Kunpeng-optimized open-source ScaNN to provide full functionality. After installing SRA_Recall, you need to:  

1. Obtain the Kunpeng-optimized open-source ScaNN code  

2. Compile the complete ScaNN Python package  

3. Generate dynamic library files  

Refer to **[Kunpeng Recall Algorithm Library](https://www.hikunpeng.com/document/detail/en/SRA/accelFeatures/recall/kunpengsra_recall_16_0030.html) Developer Guide > Installing and Configuring the Environment > Using SRA_Recall > Generating a Complete ScaNN Library** for implementation.  

### 3 Testing Guidelines

#### 3.1 Basic Dataset Overview

| Dataset       | Dimensions | Training Size | Test Size | Distance Metric | Description          |
| ------------- | ---------- | ------------- | --------- | --------------- |:-------------------- |
| GloVe         | 100        | 1183514       | 10000     | Angular         | English word vectors |
| DEEP1B        | 96         | 9990000       | 10000     | Angular         | Image vectors        |
| GIST          | 960        | 1000000       | 1000      | Euclidean       | Image GIST features  |
| SIFT          | 128        | 1000000       | 10000     | Euclidean       | Image SIFT features  |
| Fashion-MNIST | 784        | 60000         | 10000     | Euclidean       | Grayscale image data |

#### 3.2 Operational Steps

**Python API Testing:**  

Follow **[Kunpeng Recall Algorithm Library](https://www.hikunpeng.com/document/detail/en/SRA/accelFeatures/recall/kunpengsra_recall_16_0030.html) Developer Guide > KScaNN APIs > Python > Examples**  

**C++ API Testing:**  

Follow **[Kunpeng Recall Algorithm Library](https://www.hikunpeng.com/document/detail/en/SRA/accelFeatures/recall/kunpengsra_recall_16_0030.html) Developer Guide > KScaNN APIs > C++ > Examples**
