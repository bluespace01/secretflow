import secretflow as sf

# 假设有两个参与方：alice 和 bob
parties = ['alice', 'bob']

# 初始化 SecretFlow 运行时环境
sf.shutdown()
sf.init(parties, address='local')

# 定义 PYU
alice_pyu = sf.PYU('alice')
bob_pyu = sf.PYU('bob')

# cheetah_config = sf.utils.testing.cluster_def(
#     parties=['alice', 'bob'],
#     runtime_config={
#         'protocol': spu.spu_pb2.CHEETAH,
#         'field': spu.spu_pb2.FM64,
#     },
# )

# spu_device2 = sf.SPU(cheetah_config)


# 定义 Logistic Regression 模型
class LogisticRegression:
    def __init__(self, n_features):
        self.n_features = n_features
        self.weights = sf.to(alice_pyu,[0.0] * n_features)  # 初始权重
    
    def sigmoid(self, x):
        return 1 / (1 + sf.exp(-x))
    
    def predict(self, x):
        return self.sigmoid(sf.matmul(x, self.weights))
    
    def train(self, x, y, learning_rate=0.01, num_iterations=100):
        for _ in range(num_iterations):
            # 计算预测值
            y_pred = self.predict(x)
            
            # 计算梯度
            gradient = sf.dot(x.T, (y_pred - y)) / len(y)
            
            # 更新权重
            self.weights -= learning_rate * gradient
    
# 假设有一些训练数据，以及对应的标签
# 这些数据可以是私密的，因此我们将它们转换为 SPU 上的计算对象
X_train = sf.to(alice_pyu, [[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = sf.to(alice_pyu, [0, 0, 1, 1])

# 初始化 Logistic Regression 模型
model = LogisticRegression(n_features=2)

# 训练模型
model.train(X_train, y_train)

# 假设有一些测试数据
X_test = sf.to(alice_pyu, [[1, 1], [2, 2]])

# 使用模型进行预测
predictions = model.predict(X_test)

# 打印预测结果
print(predictions.reveal())
