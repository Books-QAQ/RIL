import torch
# 加载 .pth 文件
content = torch.load('//root/VIL_main13/output4/output_ril_Do_422/checkpoint/task10_checkpoint.pth')
print(content.keys()) # 查看所有键
print(content['args'])

