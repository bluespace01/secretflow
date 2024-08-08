import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 定义模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 2. 数据准备
# 这里我们用随机数据来模拟训练和测试数据
input_size = 10
hidden_size = 5
output_size = 1
num_samples = 100

X_train = torch.randn(num_samples, input_size)
y_train = torch.randint(0, 2, (num_samples, output_size)).float()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

X_test = torch.randn(10, input_size)

# 3. 训练模型
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # 二分类用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. 预测
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = (predictions > 0.5).float()
    print(predicted_classes)

# 5. 导出模型
torch.save(model.state_dict(), 'simple_nn.pt')

# 6. 加载模型
model_loaded = SimpleNN(input_size, hidden_size, output_size)
model_loaded.load_state_dict(torch.load('simple_nn.pt'))
model_loaded.eval()

# 7. 使用加载的模型进行预测
with torch.no_grad():
    predictions_loaded = model_loaded(X_test)
    predicted_classes_loaded = (predictions_loaded > 0.5).float()
    print(predicted_classes_loaded)
