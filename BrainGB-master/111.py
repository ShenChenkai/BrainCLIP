import torch
print(torch.__version__) # 查看是否包含 +cu118 这样的后缀
print(torch.version.cuda) # 查看具体的 CUDA 版本

# 再检查 pyg
import torch_geometric
print(torch_geometric.__version__)