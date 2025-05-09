import torch
import torch.nn as nn
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models"

class LearnerNN(nn.Module):
    def __init__(self, atom_params=None, atom_bias=None):
        super().__init__()
        if atom_params is not None:
            # 从线性不等式组初始化(原子命题)
            self._init_atom_nn(atom_params, atom_bias)
        else:
            raise ValueError("It is necessary to provide the config or atom_params parameters.")

    def _init_atom_nn(self, params, bias):
        """线性不等式组初始化:单神经元网络,输出sigma(Ax+b)"""
        self.input_size = len(params)
        self.hidden_size = 1
        self.output_size = 1
        self.num_layers = 1
        self.net = nn.Sequential(
            nn.Linear(len(params), 1, bias=True),
            nn.Sigmoid()
        )

        with torch.no_grad():
            self.net[0].weight.data = torch.tensor(params).float().unsqueeze(0)
            self.net[0].bias.data = torch.tensor([bias]).float()

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        return self.net(x)
    
    def predict(self, data):
        self.eval()
        with torch.no_grad():
            return self(data).item() < 0.5
        
    # def save_config(self, file_path, save_weights=True):
    #     """保存配置到JSON文件"""
    #     config = {
    #         "input_size": self.input_size,
    #         "hidden_size": self.hidden_size,
    #         "output_size": self.output_size,
    #         "num_layers": self.num_layers
    #     }
        
    #     if save_weights:
    #         # 保存模型参数
    #         config['state_dict'] = {k: v.tolist() for k, v in self.state_dict().items()}
        
    #     with open(file_path, 'w') as f:
    #         json.dump(config, f, indent=4)