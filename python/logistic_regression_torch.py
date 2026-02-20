import torch
from torch import nn
import numpy as np
import time

# Load Iris data (same as your Scala version)
def load_iris_binary():
    data = []
    with open('iris.data', 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if parts[-1] in ['Iris-setosa', 'Iris-versicolor']:
                    x0 = float(parts[0])  # sepal length
                    x1 = float(parts[1])  # sepal width
                    y = 1.0 if parts[-1] == 'Iris-versicolor' else 0.0
                    data.append([x0, x1, y])
    return np.array(data)

data = load_iris_binary()
X = torch.tensor(data[:, :2], dtype=torch.float32)
y = torch.tensor(data[:, 2], dtype=torch.float32).reshape(-1, 1)

print("=== Logistic Regression: Iris (Binary, 2 features) - PyTorch ===\n")

# Model
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        # Initialize weights to match Scala version
        with torch.no_grad():
            self.linear.weight.fill_(0.1)
            self.linear.bias.fill_(0.0)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Initial evaluation
def compute_loss_accuracy():
    with torch.no_grad():
        pred = model(X)
        loss = criterion(pred, y)
        pred_class = (pred > 0.5).float()
        accuracy = (pred_class == y).float().mean() * 100
        return loss.item(), accuracy.item()

loss, acc = compute_loss_accuracy()
print(f"Initial: loss={loss:.4f}, accuracy={acc:.2f}%\n")

# Training
start_time = time.perf_counter()
for epoch in range(100):
    # Forward pass
    pred = model(X)
    loss = criterion(pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        loss_val, acc = compute_loss_accuracy()
        print(f"Epoch {epoch:3d}: loss={loss_val:.4f}, accuracy={acc:.2f}%")

training_time = (time.perf_counter() - start_time) * 1000  # ms

loss, acc = compute_loss_accuracy()
print(f"\nFinal: loss={loss:.4f}, accuracy={acc:.2f}%")
w0, w1 = model.linear.weight[0]
bias = model.linear.bias[0]
print(f"Learned weights: w0={w0:.4f}, w1={w1:.4f}, bias={bias:.4f}")
print(f"\nTraining time: {training_time:.2f} ms")

# Inference benchmark
print("\n=== Inference Benchmark ===")
n_iterations = 10000
test_input = torch.tensor([[5.1, 3.5]], dtype=torch.float32)

# Warmup
for _ in range(100):
    with torch.no_grad():
        _ = model(test_input)

# Benchmark
start = time.perf_counter()
with torch.no_grad():
    for _ in range(n_iterations):
        _ = model(test_input)
end = time.perf_counter()

inference_time_us = (end - start) / n_iterations * 1e6
print(f"Average inference time: {inference_time_us:.2f} μs per prediction")
print(f"Total for {n_iterations} predictions: {(end - start) * 1000:.2f} ms")

