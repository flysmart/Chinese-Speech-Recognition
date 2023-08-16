import torch

# 加载checkpoint.tar文件
checkpoint = torch.load('BEST_checkpoint.tar')

# 从checkpoint中提取model对象
model = checkpoint['model']

# 保存model对象为model.pt文件
torch.save(model.state_dict(), 'model.pt')
