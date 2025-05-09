import torch
import torch.nn as nn
from z3 import *

model_path = "models"  # 模型保存路径

class PAModel:
    def __init__(self, vars, formula_str, model=None, pos_data=None, neg_data=None):

        self.vars = vars
        self.z3_vars = {v: Int(v) for v in vars}
        self.z3_formula = eval(formula_str, {'__builtins__': None}, self.z3_vars)

        self.pos_data = pos_data if pos_data else None
        self.neg_data = neg_data if neg_data else None
        if model:
            self.model = model

    def _init_from_path(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} does not exist.")
        
        self.model = torch.load(path)

    def train(self, epochs=100):

        if self.pos_data is None or self.neg_data is None:
            raise ValueError("Training data not provided")
        
        X = torch.cat([self.pos_data, self.neg_data])
        y = torch.cat([
            torch.ones(len(self.pos_data)), 
            torch.zeros(len(self.neg_data))
        ]).unsqueeze(1)
        
        # 训练过程
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = nn.BCELoss()
        
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

    def predict(self, inputs):
        """使用神经网络预测"""
        if not all(isinstance(i, int) for i in inputs):
            raise ValueError("Inputs must be integers")

        return self.model.predict(inputs)
        
    def verify(self, inputs):
        """使用逻辑技术判断"""
        if not all(isinstance(i, int) for i in inputs):
            raise ValueError("Inputs must be integers")
            
        s = Solver()
        s.add(self.z3_formula)
        for var, val in zip(self.vars, inputs):
            s.add(self.z3_vars[var] == val)  # 关键点3：直接使用整数比较
        return s.check() == sat
    
    def add_example(self, example, is_positive=True):
        """添加数据样例"""
        tensor = torch.tensor(example, dtype=torch.float32)
        if is_positive:
            self.positive_examples.append(tensor)
        else:
            self.negative_examples.append(tensor)

    def update(self, new_pos_data, new_neg_data):
        """
        合并新数据并重新训练模型
        :param new_pos_data: 新增正例集（整数列表/张量）
        :param new_neg_data: 新增反例集（整数列表/张量）
        """
        # 验证并转换新数据为整数张量
        new_pos = self._validate_int_data(new_pos_data) if new_pos_data else None
        new_neg = self._validate_int_data(new_neg_data) if new_neg_data else None
        
        # 合并数据（处理初始为空的情况）
        self.pos_data = torch.cat([self.pos_data, new_pos]) if self.pos_data is not None else new_pos
        self.neg_data = torch.cat([self.neg_data, new_neg]) if self.neg_data is not None else new_neg
        
        # 重新训练模型
        self.train()

        # TODO: 训练策略改为增量学习

    def save_config(self, file_path, save_weights=True):
        file_path = f"{model_path}/{file_path}"
        if save_weights:
            # 保存模型权重
            torch.save(self.model, file_path)
        else:
            torch.save(self.model.state_dict(), file_path)

# def combine_presburger_models(M1, M2):
#     """
#     合并两个PresburgerModel对象，处理共享变量y,z
#     :param M1: 含变量x,y,z的模型
#     :param M2: 含变量y,z,w的模型
#     :return: 含变量x,y,z,w的新模型M3
#     """
#     # 合并变量（保持共享变量y,z的一致性）
#     new_vars = ['x'] + sorted(list(set(M1.variables) & set(M2.variables))) + ['w']  # ['x','y','z','w']
    
#     # 构建新公式（示例使用逻辑与组合，实际可根据需求修改）
#     new_formula = f"And({M1.formula_str}, {M2.formula_str})"
    
#     # 构建集成神经网络（使用PyTorch的ModuleList）
#     class IntegratedModel(nn.Module):
#         def __init__(self, model1, model2):
#             super().__init__()
#             self.model1 = model1.model
#             self.model2 = model2.model
#             self.combine = nn.Linear(2, 1)  # 集成两个模型的输出
            
#         def forward(self, x):
#             # 分割输入：x[0]对应x, x[1:3]对应y,z, x[3]对应w
#             out1 = self.model1(x[[0,1,2]])  # M1处理x,y,z
#             out2 = self.model2(x[[1,2,3]])  # M2处理y,z,w
#             return torch.sigmoid(self.combine(torch.cat([out1, out2], dim=1)))
    
#     # 创建新模型实例
#     integrated_nn = IntegratedModel(M1, M2)
#     return PAModel(integrated_nn, new_formula, new_vars)

if __name__ == "__main__":
    # 定义整数变量和公式
    vars = ['x', 'y', 'z']
    formula = "x + y == z"

    # 整数训练数据
    pos_data = [[1,2,3], [0,0,0]]  # 满足公式的整数示例
    neg_data = [[1,1,3], [2,2,5]]  # 不满足的整数示例

    # 创建并训练模型
    model = PAModel(vars, formula, pos_data, neg_data)
    model.train(epochs=50)

    # 测试（必须输入整数）
    print(model.predict([2,3,5]))  # 输出1（满足）
    print(model.verify([2,3,5]))   # 输出True