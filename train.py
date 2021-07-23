import glob
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
#from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
#from tqdm.notebook import tqdm

from repmlp.repmlp_resnet import *
from utils import accuracy, ProgressMeter, AverageMeter, load_checkpoint
from AdMSLoss import AdMSoftmaxLoss

# checkpoint
PRETRAINED = '' #'../face_model/RepMLP/RepMLP-Res50-light-224_train.pth'
CHECKPOINT = 'data/checkpoint.pt'
total_epochs = 0



# 训练图片路径
train_dir = 'data/train'
val_dir = 'data/val'


# Training settings
batch_size = 128
epochs = 20
lr = 0.001
gamma = 0.7
seed = 42
img_size = 96 # 224

print(f"batch_size={batch_size}, epochs={epochs}, lr={lr}")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


#device = 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

os.makedirs('data', exist_ok=True)

# 准备 labels
label_list = os.listdir(train_dir)
label_dict = { v:i for i,v in enumerate(label_list)}
print(f"num_classes: {len(label_list)}")

# load data
train_list = glob.glob(os.path.join(train_dir+'/*','*.jpg'))
valid_list = glob.glob(os.path.join(val_dir+'/*','*.jpg'))

labels = [path.split('/')[-2] for path in train_list]

# split
#train_list, valid_list = train_test_split(train_list, 
#    test_size=0.2, 
#    stratify=labels, # 对有些数据集需要注释掉，标签下只有1个数据的情况
#    random_state=seed)

print(f"Train Set: {len(train_list)}")
print(f"Validation Set: {len(valid_list)}")


# Image Augumentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        #transforms.RandomResizedCrop(img_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        #transforms.RandomResizedCrop(img_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]
)


# Load Datasets

class FaceDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-2]

        return img_transformed, label_dict[label]

train_data = FaceDataset(train_list, transform=train_transforms)
valid_data = FaceDataset(valid_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)


# model
#model = create_RepMLPRes50_Light_224(deploy=False)
model = RepMLPResNet(num_blocks=[3,4,6,3], num_classes=len(label_list), 
            block_type='bottleneck', img_H=img_size, img_W=img_size,
            h=6, w=6, reparam_conv_k=(1,3,5), fc1_fc2_reduction=1, fc3_groups=4, 
            deploy=False, bottleneck_r=(2,4))

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

if not torch.cuda.is_available():
    print('using CPU, this will be slow')
    use_gpu = False
else:
    model = model.cuda()
    use_gpu = True

# Training


# loss function
criterion = nn.CrossEntropyLoss()
#in_features = len(label_list)
#out_features = len(label_list) # Number of classes
#criterion = AdMSoftmaxLoss(in_features, out_features, s=30.0, m=0.4).to(device) # Default values recommended by [1]
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


if os.path.exists(CHECKPOINT):
    checkpoint = torch.load(CHECKPOINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    total_epochs = checkpoint['epoch']
    last_loss = checkpoint['loss']
    last_label = checkpoint['label_dict']
    print(f"Loaded {CHECKPOINT}: epochs= {total_epochs}, loss= {last_loss:.6f}, num_classes= {len(last_label)}")
else:
    # 载入 pre-train checkpoint
    if os.path.exists(PRETRAINED):
        print("=> loading pre-trained checkpoint '{}'".format(PRETRAINED))
        load_checkpoint(model, PRETRAINED)


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - lr: {scheduler.get_last_lr()[0]}\n"
    )

    #scheduler.step()


# 保存
torch.save({
            'epoch'                : total_epochs+epochs,
            'model_state_dict'     : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss'                 : epoch_loss,
            'label_dict'           : label_dict,
            }, CHECKPOINT)


'''
# 清除训练时缓存， 用在notebook时

model = None
torch.cuda.empty_cache()

import gc
gc.collect()
'''