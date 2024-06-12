import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision as tv

import torch.utils.data as data

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from torch.cuda.amp import autocast, GradScaler


class CustomDataset(data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.list_dir1 = sorted(os.listdir(path_dir1))
        self.list_dir2 = sorted(os.listdir(path_dir2))

    def __len__(self):
        return len(self.list_dir1) + len(self.list_dir2)

    def __getitem__(self, idx):
        if idx < len(self.list_dir1):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.list_dir1[idx])
        else:
            class_id = 1
            img_path = os.path.join(self.path_dir2, self.list_dir2[idx - len(self.list_dir1)])
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = img / 255.0

            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            img = img.transpose((2, 0, 1))

            t_img = torch.from_numpy(img)
            t_class_id = torch.tensor(class_id, dtype=torch.long)

        except Exception as e:
            print(img_path, 'dolboebskii', e)
            return

        return {'img': t_img, 'class_id': t_class_id}


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm0 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm1(out)
        return self.act(out + x)


class BottleNeck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv0 = nn.Conv2d(channels, channels // 4, kernel_size=1, padding=0)
        self.norm0 = nn.BatchNorm2d(channels // 4)
        self.conv1 = nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels // 4)
        self.conv2 = nn.Conv2d(channels // 4, channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out + x
        # return self.act(out + x)


class ResSequence(nn.Module):
    def __init__(self, channels, seq_len, block_type='classic'):
        super().__init__()

        seq = []
        for i in range(seq_len):
            if block_type == 'classic':
                seq.append(ResBlock(channels))
            elif block_type == 'bottleneck':
                seq.append(BottleNeck(channels))
            else:
                raise NotImplementedError(f"Block type {block_type} not implemented")
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, channels, kernel_size=7, stride=2)
        self.norm0 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.layer1 = ResSequence(channels, 3, block_type='bottleneck')
        self.conv1 = nn.Conv2d(channels, 2*channels, kernel_size=3, padding=1)
        self.layer2 = ResSequence(2*channels, 4, block_type='bottleneck')
        self.conv2 = nn.Conv2d(2*channels, 4*channels, kernel_size=3, padding=1, stride=2)
        self.layer3 = ResSequence(4*channels, 6, block_type='bottleneck')
        self.conv3 = nn.Conv2d(4*channels, 4*channels, kernel_size=3, padding=1, stride=2)
        self.layer4 = ResSequence(4*channels, 3, block_type='bottleneck')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4*channels, out_channels)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.conv1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        out = self.layer3(out)
        out = self.conv3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_cats_path = './train/cats'
train_dogs_path = './train/dogs'
test_cats_path = './test/cats'
test_dogs_path = './test/dogs'

train_ds = CustomDataset(train_cats_path, train_dogs_path)
test_ds = CustomDataset(test_cats_path, test_dogs_path)

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_ds, shuffle=True,
    batch_size=batch_size, num_workers=1, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds, shuffle=True,
    batch_size=batch_size, num_workers=1, drop_last=False
)

model = ResNet(3, 64, 2)

print(count_parameters(model))


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


def accuracy(pred, label):
    answer = F.softmax(pred.detach(), dim=1).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(DEVICE)
loss_func = loss_func.to(DEVICE)

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

EPOCHS = 10
loss_epochs_list = []
accuracy_epochs_list = []
for epoch in range(EPOCHS):
    loss_val = 0
    acc_val = 0
    for sample in (pbar := tqdm(train_loader)):
        img, label = sample['img'], sample['class_id']
        label = F.one_hot(label, 2).float()
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()

        with autocast(use_amp):
            pred = model(img)
            loss = loss_func(pred, label)

        scaler.scale(loss).backward()
        loss_item = loss.item()
        loss_val += loss_item

        scaler.step(optimizer)
        scaler.update()

        acc_current = accuracy(pred.cpu().float(), label.cpu().float())
        acc_val += acc_current

        pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
    scheduler.step()
    loss_epochs_list += [loss_val / len(train_loader)]
    accuracy_epochs_list += [acc_val / len(train_loader)]
    print(loss_epochs_list[-1])
    print(accuracy_epochs_list[-1])
