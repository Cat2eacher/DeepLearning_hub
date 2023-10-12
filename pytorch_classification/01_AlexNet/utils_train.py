import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

'''
/****************************************************/
获得学习率
/****************************************************/
'''


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


'''
/****************************************************/
获得学习率
/****************************************************/
'''


# ----------------------------------------------------#
#   训练
# ----------------------------------------------------#
def train_epoch(model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches):
    total_loss = 0.0
    total_accuracy = 0.0
    model.train()
    train_bar = tqdm(train_dataloader, total=len(train_dataloader))
    # for index, (inputs, targets) in enumerate(train_dataloader, start=1):
    for index, (inputs, targets) in enumerate(train_bar, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicts = torch.max(outputs, dim=1)
        num_correct = (predicts == targets).sum()
        accuracy = float(num_correct) / float(inputs.shape[0])
        total_accuracy += accuracy
        # accuracy = torch.mean(
        #     (torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
        # total_accuracy += accuracy.item()

        train_bar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')
        train_bar.set_postfix(**{'total_loss': total_loss / index,
                                 'accuracy': total_accuracy / index,
                                 'lr': get_lr(optimizer),
                                 })

    return total_accuracy / index


# ----------------------------------------------------#
#   验证
# ----------------------------------------------------#
def valid_epoch(model, device, valid_dataloader, criterion, epoch):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
        # for index, (inputs, targets) in enumerate(train_dataloader, start=1):
        for index, (inputs, targets) in enumerate(valid_bar, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            # loss = criterion(outputs, targets)
            # total_loss += loss.item()

            accuracy = torch.mean(
                (torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

            val_accurate = total_accuracy / index
            valid_bar.set_description('valid')
            valid_bar.set_postfix(**{'accuracy': val_accurate,
                                     })
    print('[epoch %d]  val_accuracy: %.3f' %
          (epoch + 1,  val_accurate))
    return val_accurate


def checkpoint(model, optimizer, epoch, scheduler, checkpoint_path):
    checkpoints = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch,
                   'lr': scheduler.state_dict(),
                   }
    save_path = checkpoint_path
    torch.save(checkpoints, save_path)
