import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Current Device: {device}")

class AtomicNetwork(nn.Module):
    def __init__(self, coefficients, bias):
        super().__init__()
        self.linear = nn.Linear(len(coefficients), 1, bias=True)

        with torch.no_grad():
            self.linear.weight.data = torch.tensor([coefficients], dtype=torch.float32)
            self.linear.bias.data.fill_(-bias)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        return torch.sigmoid(self.linear(x))
    
    def predict(self, data):
        self.eval()
        with torch.no_grad():
            return self(data).item() < 0.5

if __name__ == "__main__":
    model = AtomicNetwork([2, 3, -5], 1).to(device) # 2x + 3y - 5z < 1
    test_data = torch.tensor([[1, 1, 1]])
    print("Predicted results:", model.predict(test_data))