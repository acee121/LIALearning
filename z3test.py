from z3 import *
import torch
from utils import z3tools
from LearnerNN import LearnerNN
from PAModel import PAModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")

data_1 = [
    [[1, 1, 1], [0, 0, 0], [1, 1, 4], [2, 2, 5]],
    [0,0,1,1]
]

model_1 = LearnerNN([1, 1, -1], 0).to(device)
Learner_1 = PAModel("linear1.smt2", model_1, data_1)

for i in range(10):
        print(f"{i}:"+str()+str(Learner_1.predict([2, 2, i]))+" "+str(Learner_1.verify({'x':2, 'y':2, 'z':i})))