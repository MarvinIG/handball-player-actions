import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from contentBox import extract_red_region, extract_red_region_and_remove_black

###########################################################
# positions

class CNN_Positions_v1(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Positions_v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 39 * 39, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 156x156 -> 78x78
        x = self.pool(F.relu(self.conv2(x)))  # 78x78 -> 39x39
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

CNN_train_transforms_v1 = transforms.Compose([
    transforms.Resize((156, 156)),
    transforms.ToTensor(),
])

CNN_val_transforms_v1 = transforms.Compose([
    transforms.Resize((156, 156)),
    transforms.ToTensor(),
])


class CNN_Positions_v2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Positions_v2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 64 * 64, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

CNN_train_transforms_v2 = transforms.Compose([
    transforms.Resize((300, 300)),         
    transforms.RandomRotation(10),          
    transforms.CenterCrop((256, 256)),      
    transforms.ToTensor(),
])

CNN_val_transforms_v2 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop((256, 256)),      # Crop to the same target size for validation
    transforms.ToTensor(),
])


class CNN_Positions_V3(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Positions_V3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(51200, 256)  
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        x = x.view(x.size(0), -1)
        # print(f"After flattening: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

CNN_train_transforms_Positions_V3 = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

CNN_val_transforms_Positions_V3 = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

CNN_train_transforms_Positions_V3b = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomRotation(10),          
    transforms.CenterCrop((270, 270)), 
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

###########################################################
# shotTypes
class CNN_shotType_V1(nn.Module):
    def __init__(self, num_classes):
        super(CNN_shotType_V1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(97344, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

CNN_train_transforms_shotType_V1 = transforms.Compose([
    transforms.Resize((156, 156)),
    transforms.Lambda(lambda img: extract_red_region(img, output_size=(156, 156))),
    transforms.ToTensor(),
])

CNN_train_transforms_shotType_V1B = transforms.Compose([
    transforms.Resize((156, 156)),
    transforms.Lambda(lambda img: extract_red_region_and_remove_black(img, output_size=(156, 156))),
    transforms.ToTensor(),
])

CNN_val_transforms_shotType_V1B = transforms.Compose([
    transforms.Resize((156, 156)),
    transforms.Lambda(lambda img: extract_red_region_and_remove_black(img, output_size=(156, 156))),
    transforms.ToTensor(),
])

CNN_val_transforms_shotType_V1 = transforms.Compose([
    transforms.Resize((156, 156)),
    transforms.Lambda(lambda img: extract_red_region(img, output_size=(156, 156))),
    transforms.ToTensor(),
])


class CNN_shotType_V2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_shotType_V2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 40 * 40, 256)  
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

CNN_train_transforms_shotType_V2 = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.Lambda(lambda img: extract_red_region(img, output_size=(320, 320))),
    transforms.ToTensor(),
])

CNN_val_transforms_shotType_V2 = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.Lambda(lambda img: extract_red_region(img, output_size=(320, 320))),
    transforms.ToTensor(),
])

class CNN_shotType_Binary(nn.Module):
    def __init__(self, *args):
        super(CNN_shotType_Binary, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)  
        self.fc1 = nn.Linear(204800, 32)  
        self.fc2 = nn.Linear(32, 1) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 156x156 -> 78x78
        x = self.pool(F.relu(self.conv2(x)))  # 78x78 -> 39x39
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output is a single logit for binary classification
        return x
    
CNN_train_transforms_shotType_Binary_V2 = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.Lambda(lambda img: extract_red_region_and_remove_black(img, output_size=(320, 320))),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(320, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

###########################################################
# trickShots

###########################################################
# feints
class CNN_feints_V1(nn.Module):
    MODELSIZE = (156,156)
    MODELNAME = 'feints_V1'
    def __init__(self, num_classes):
        super(CNN_feints_V1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(97344, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 156x156 -> 78x78
        x = self.pool(F.relu(self.conv2(x)))  # 78x78 -> 39x39
        # print(f"After conv2 + pool: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"After flattening: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

CNN_train_transforms_feints_V1B = transforms.Compose([
    transforms.Resize((156, 156)),
    transforms.Lambda(lambda img: extract_red_region_and_remove_black(img, output_size=(156, 156))),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

CNN_val_transforms_feints_V1B = transforms.Compose([
    transforms.Resize((156, 156)),
    transforms.Lambda(lambda img: extract_red_region_and_remove_black(img, output_size=(156, 156))),
    transforms.ToTensor(),
])

def transformerAttributes(size: tuple, extractRed: int = 0, randomRot: int = None, randomFlip: bool = False):
    transforms_list = []
    transforms_list.append(transforms.Resize(size))
    match extractRed:
        case 0:
            pass
        case 1:
            transforms_list.append(transforms.Lambda(lambda img: extract_red_region(img, output_size=size)))
        case 2:
            transforms_list.append(transforms.Lambda(lambda img: extract_red_region_and_remove_black(img, output_size=size)))
    if randomRot:
        transforms_list.append(transforms.RandomRotation(randomRot))
    if randomFlip:
        transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)

###########################################################
# blocks
class CNN_blocks_V1(nn.Module):
    MODELSIZE = (320,320)
    MODELNAME = 'blocks_V1'
    def __init__(self, num_classes):
        super(CNN_blocks_V1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(102400, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        # print(f"After conv2 + pool: {x.shape}")
        x = x.view(x.size(0), -1)
        #print(f"After flattening: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN_blocks_V3(nn.Module):
    MODELSIZE = (320,320)
    MODELNAME = 'blocks_V3'
    def __init__(self, num_classes):
        super(CNN_blocks_V3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(51200, 256)  
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        x = x.view(x.size(0), -1)
        # print(f"After flattening: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

