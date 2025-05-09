import torch
import torch.nn as nn
from LearnerNN import LearnerNN
from PAModel import PAModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")
model_path = "models"

if __name__ == "__main__":

    # 定义原子命题的网络
    vars_1 = ['x', 'y', 'z']
    formula_1 = "2*x + 3*y - 5*z < 1"

    pos_1 = [[1, 1, 1], [0, 0, 0]]
    neg_1 = [[1, 1, 2], [2, 2, 5]]

    model_1 = LearnerNN([2, 3, -5], -1).to(device)
    loaded_params = torch.load(f"{model_path}/atom_model_1.pth")
    model_1.load_state_dict(loaded_params)
    Learner_1 = PAModel(vars_1, formula_1, model_1, pos_1, neg_1)
    # Learner_1.save_config("atom_model_1.pth", save_weights=False)

    for i in range(10):
        print(f"{i}:"+str()+str(Learner_1.predict([2, 2, i]))+" "+str(Learner_1.verify([2, 2, i])))

    # 定义另一个原子命题的网络
    vars_2 = ['y', 'z', 'w']
    formula_2 = "y + z > w"

    pos_2 = [[1, 1, 1], [2, 6, 3]]
    neg_2 = [[1, 1, 2], [2, 2, 5]]

    model_2 = LearnerNN([-1, -1, 1], 0).to(device)
    Learner_2 = PAModel(vars_2, formula_2, model_2, pos_2, neg_2)

    # for i in range(10):
    #     print(f"{i}:"+str()+str(Learner_2.predict([2, 4, i]))+" "+str(Learner_2.verify([2, 4, i])))

    # 定义"逻辑与"网络
    vars_3 = ['x', 'y', 'z', 'w']
    formula_3 = "And(2*x + 3*y - 5*z < 1, y + z > w)"
    
    # TODO: 在LearnerNN.py里添加一个逻辑与方法, 接收两个网络输出逻辑与网络。初步想法：ensemble knowledge distillation