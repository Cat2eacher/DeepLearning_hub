import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from model import AlexNet
from utils_train import train_epoch, valid_epoch, checkpoint
'''
/**************************task1**************************/
数据集预处理
/**************************task1**************************/
'''
# ----------------------------------------------------#
#           数据集路径
# ----------------------------------------------------#

dataset_dir = "../CatVsDog"
assert os.path.exists(dataset_dir), "{} path does not exist.".format(dataset_dir)
# train_dir = data_dir + '/train'
# valid_dir = data_dir + '/valid'
# test_dir = data_dir + '/test'

# ----------------------------------------------------#
#           data_transforms
# ----------------------------------------------------#
data_transforms = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ]),
    "valid": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                 transforms.ToTensor(),
                                 ]),
}

# ----------------------------------------------------#
#           dataset, dataloader
# ----------------------------------------------------#
batch_size = 8
# num_worker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 2])  # number of workers

image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
print("using {} images for training, {} images for validation.".format(dataset_sizes['train'],
                                                                       dataset_sizes['valid']))
# class_list = image_datasets['train'].class_to_idx
# train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'),
#                                      transform=data_transforms["train"])
# train_num = len(train_dataset)

'''
/**************************task2**************************/
设备device
/**************************task2**************************/
'''
# ----------------------------------------------------#
# device
# ----------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

'''
/**************************task3**************************/
网络模型
/**************************task3**************************/
'''
# ----------------------------------------------------#
#   创建模型
# ----------------------------------------------------#
model = AlexNet(num_classes=2)

# ----------------------------------------------------#
#   导入模型参数
# ----------------------------------------------------#
model_path = ""

if model_path != "":
    # 载入与训练权重
    print('Loading weights into state dict...')
    # 读取当前模型参数
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # 更新当前模型参数
    model_dict.update(pretrained_dict)
    # 加载模型参数
    model.load_state_dict(model_dict)

# ----------------------------------------------------#
#   模型部署
# ----------------------------------------------------#
model.to(device)

'''
/**************************task4**************************/
训练
/**************************task4**************************/
'''
# 损失函数
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

# 学习率和优化策略
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)  # 学习率每1个epoch衰减成原来的0.94

num_epochs = 40
save_path = 'AlexNet.pth'
best_acc = 0.0

for epoch in range(num_epochs):
    # train
    train_accurate = train_epoch(model, device, dataloaders["train"], criterion, optimizer, epoch, num_epochs)
    # update LR
    # print("第%d轮epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
    lr_scheduler.step()
    # validate
    valid_accurate = valid_epoch(model, device, dataloaders["valid"], criterion)

    # checkpoint
    if valid_accurate > best_acc:
        best_acc = valid_accurate
        torch.save(model.state_dict(), save_path)

print('Finished Training')
