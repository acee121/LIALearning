import torch
import torch.nn as nn
from z3 import *
from utils import z3tools

formula_path = "formulas"
model_path = "models"
data_path = "datasets"

class PAModel:
    def __init__(self, formula_file, model=None, data=None):

        self.formula_file = f"{formula_path}/{formula_file}"
        self.solver = Solver()
        print(f"---Loading formula from {self.formula_file}---")
        self.solver.from_file(self.formula_file)
        self.vars = z3tools.get_variables(self.solver)
        self.data = data

        if model:
            self.model = model

    def _init_from_path(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} does not exist.")
        
        self.model = torch.load(path)

    def train(self, epochs=100):
        if self.data is None:
            raise ValueError("Training data not provided")
        
        X = self.data[0]
        y = self.data[1]

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
        """
        检查输入的整数向量是否满足当前求解器的约束条件
        :param inputs: 字典形式，例如 {'x': 1, 'y': 2}
        :return: True 如果满足约束，否则 False
        """
        # 创建临时求解器用于验证（避免污染原求解器状态）
        temp_solver = Solver()
        for assertion in self.solver.assertions():
            temp_solver.add(assertion)
        
        for var_name, value in inputs.items():
            # 检查变量是否在集合中
            var = next((v for v in self.vars if str(v) == var_name), None)
            if var is not None:
                temp_solver.add(var == value)
            else:
                raise ValueError(f"变量 {var_name} 不在模型中")
        
        return temp_solver.check() == sat
    
    def add_example(self, example, is_positive=True):
        """添加数据样例"""
        tensor = torch.tensor(example, dtype=torch.float32)
        if is_positive:
            self.positive_examples.append(tensor)
        else:
            self.negative_examples.append(tensor)

    def update(self, new_data):
        
        # 合并数据（处理初始为空的情况）
        self.data[0] = torch.cat([self.data[0], new_data[0]]) if self.data is not None else new_data[0]
        self.data[1] = torch.cat([self.data[1], new_data[1]]) if self.data is not None else new_data[1]
        
        # 重新训练模型
        self.train()

        # TODO: 训练策略改为增量学习

    def save_config(self, file_path=None):
        if file_path is None:
            file_path = f"{model_path}/{self.formula_str}.pth"
        file_path = f"{model_path}/{file_path}"
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