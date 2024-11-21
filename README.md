# sra_scann_adapter

### 1 介绍

Adapter for Kunpeng ScaNN Library

### 2 使用说明

#### 2.1 安装SRA_Recall

请参照《鲲鹏召回算法库 开发指南》（链接待更新）2.3.1章节进行操作。

#### 2.2 生成完整的ScaNN

KScaNN依赖于鲲鹏开源的ScaNN以提供完整的功能。因此安装SRA_Recall后需利用本仓库所提供鲲鹏开源的ScaNN源代码，编译出完整的ScaNN安装包。

请参照《鲲鹏召回算法库 开发指南》（链接待更新）2.3.2章节进行操作。

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

**步骤1：**

在2.2章节中已下载本仓库上的源码至测试服务器可访问的路径中，假设位于`/path/to/scann/sra_scann_adapter`，测试框架位于`/path/to/scann/sra_scann_adapter/ann-benchmarks`下。

**步骤2：**

下载基础数据集至`/path/to/scann/sra_scann_adapter/ann-benchmarks/data`下。

```
cd /path/to/scann/sra_scann_adapter/ann-benchmarks/
mkdir data
cd data
wget http://ann-benchmarks.com/glove-100-angular.hdf5 --no-check-certficate
wget http://ann-benchmarks.com/deep-image-96-angular.hdf5 --no-check-certficate
wget http://ann-benchmarks.com/gist-960-euclidean.hdf5 --no-check-certficate
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 --no-check-certficate
wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5 --no-check-certficate
```

**步骤3：**

安装numactl依赖。

```
yum install numactl
```

**步骤4：**

根据测试服务器numa数量及分布，修改`/path/to/scann/sra_scann_adapter/ann-benchmarks/test.sh`中198~201行的内容：

+ 若numa数量为4，无需修改；

+ 若numa数量不为4，此处假设为8，可按以下方式修改：
  
  ```
  time numactl -N 0 -m 0 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
  time numactl -N 1 -m 1 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
  time numactl -N 2 -m 2 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
  time numactl -N 3 -m 3 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
  time numactl -N 4 -m 4 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
  time numactl -N 5 -m 5 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
  time numactl -N 6 -m 6 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
  time numactl -N 7 -m 7 -- python run.py --dataset ${data} --algorithm scann --local --force --batch --runs 8&
  ```

**步骤5：**

执行测试。

```
cd /path/to/scann/sra_scann_adapter/ann-benchmarks/
sh test.sh
```
