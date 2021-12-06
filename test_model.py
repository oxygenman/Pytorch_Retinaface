import torch
model=torch.load("retinaface.pth")
print(model)
weight_dict=model.state_dict()
for name,value in weight_dict.items():
    print(name)
print(weight_dict["body.stage1.0.0.weight"])
