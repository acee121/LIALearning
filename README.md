# LIALearning

## 环境需求

```txt
python
torch
z3_solver
```

均为最新版本，torch需要gpu。

## 项目结构

### 神经网络类 LearnerNN

LearnerNN.py 中的 `LearnerNN` 类，定义了一个二分类神经网络。为观察其结构，有四个变量：

```python
self.input_size   # 输入维数（变量个数）
self.hidden_size  # 隐藏层大小
self.output_size  # 输出维数（实际上就是2）
self.num_layers   # 层数
```

两种初始化方法：

* 如果要从json文件构建神经网络，则只需指定一个参数`config`，传入json文件的路径；
* 如果要构建原子命题对应的网络，则将第一个参数设置为`config = None`，然后传入一个矩阵和一个偏置。这些都是整数。

输出模型参数的方法：`save_config(file_path, save_weights=True)`

### 学习器类 PAModel

PAModel.py 中的 `PAModel` 类，定义了连同神经网络在内的整个学习模型。包括：

```python
self.vars       # 变量名列表 (如 ['x','y','z'])
self.z3_vars    # 转化得到的z3变量，自动初始化，不需传入
self.z3_formula # 逻辑公式字符串 (如 "x + y == z")
self.pos_data   # 正例数据集（整数张量）
self.neg_data   # 反例数据集（整数张量）
self.model      # 神经网络
```

初始化时，需要传入以上这些变量。其中主要包含的方法有：

* `predict(inputs)`：给定整数数组`inputs`，返回神经网络的输出（预测解）
* `verify(inputs)`：给定整数数组`inputs`，返回z3求解器的输出（正确解）
* `update(new_pos_data, new_neg_data)`：给定新增的正反例数据，更新数据集并重新训练

