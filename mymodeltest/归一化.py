import torch

# 模拟嵌入向量
embedding = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
eps = 1e-8

# 计算RMS
rms = torch.sqrt(torch.mean(embedding**2, dim=1, keepdim=True) + eps)
print(rms)
# 进行RMS归一化
normalized_embedding = embedding / rms

print(normalized_embedding)
