import os
import random
from z3 import Int, Solver, sat
import pandas as pd
from typing import Dict, List

class Z3DataGenerator:
    def __init__(self):
        """初始化求解器和默认参数"""
        self.solver = Solver()
        self.vars = ['x', 'y', 'z']  # 默认变量名（可通过代码直接修改）
        self.z3_vars = {v: Int(v) for v in self.vars}
        self.formula_str = "[x + y > 3, z < 10]"  # 默认公式（可直接修改）
        self.n_samples = 100  # 默认采样数
        self.max_val = 100  # 默认变量最大值
        
        # 解析公式（需先设置formula_str）
        self._parse_formula()

    def _parse_formula(self):
        """解析Z3公式"""
        try:
            print(f"解析公式: {self.formula_str}")
            self.z3_formula = eval(
                self.formula_str,
                {'__builtins__': None},
                self.z3_vars
            )
            self.solver.add(self.z3_formula)
        except Exception as e:
            raise ValueError(f"公式解析失败: {e}")

    def _generate_sample(self) -> Dict[str, int]:
        """生成随机样本"""
        return {v: random.randint(-self.max_val, self.max_val) for v in self.vars}

    def _check_sample(self, sample: Dict[str, int]) -> bool:
        """验证样本是否满足公式"""
        temp_solver = Solver()
        temp_solver.add(self.z3_formula)
        for v in self.vars:
            temp_solver.add(self.z3_vars[v] == sample[v])
        return temp_solver.check() == sat

    def generate_data(self) -> pd.DataFrame:
        """生成数据集（不保证样本均衡）"""
        data = []
        while len(data) < self.n_samples:
            sample = self._generate_sample()
            sample['label'] = 1 if self._check_sample(sample) else 0
            data.append(sample)
        return pd.DataFrame(data)

    def save_data(self, df: pd.DataFrame, dataset_dir: str = "datasets"):
        """保存数据集到CSV"""
        os.makedirs(dataset_dir, exist_ok=True)
        filepath = os.path.join(dataset_dir, "z3_data.csv")
        df.to_csv(filepath, index=False)
        print(f"数据集已保存到: {filepath}")

# 使用示例
if __name__ == "__main__":
    generator = Z3DataGenerator()
    
    # 直接修改参数（无需外部传入）
    generator.formula_str = "[x + y > 3, z < 10]"  # 修改公式
    generator.vars = ['x', 'y', 'z']  # 修改变量名
    generator.n_samples = 100  # 修改采样数
    generator.max_val = 100  # 修改变量范围
    
    # 生成并保存数据
    df = generator.generate_data()
    generator.save_data(df)