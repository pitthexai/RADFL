import numpy as np 
import pandas as pd
import math
import os
from PIL import Image
from tqdm import tqdm
import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import transforms
from monai.networks.nets import ResNetFeatures, ViT, resnet18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Creating a Transformation Object
transforms = v2.Compose([
    #Converting images to the size that the model expects
    v2.Resize(size=(224,224)),
    v2.ToTensor(), #Converting to tensor
    v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #Normalizing the data to the data that the ResNet18 was trained on
    
])


# Create custom dataset class
class XrayDataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
    
        if self.transforms:
            img = self.transforms(img)

        return img, label
    
cv_best_models = []
cv_best_acc = []
batch_size = 32
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define dataset paths
train_image_paths = []
train_labels = []
for label in [0, 1]:
    folder = f"/home/feg48/fl_xray_project/input_up_baseline/training/{label}"
    for img_path in os.listdir(folder):
        train_image_paths.append(os.path.join(folder, img_path))
        train_labels.append(label)

xray_train_dataset = XrayDataset(train_image_paths, train_labels, transforms=transforms)
xray_train_dataloader = DataLoader(xray_train_dataset, batch_size=batch_size, shuffle=True)

# Define testing dataset paths
test_image_paths = []
test_labels = []
for label in [0, 1]:
    folder = f"/home/feg48/fl_xray_project/input_up_baseline/testing/{label}"
    for img_path in os.listdir(folder):
        test_image_paths.append(os.path.join(folder, img_path))
        test_labels.append(label)

xray_test_dataset = XrayDataset(test_image_paths, test_labels, transforms=transforms)
xray_test_dataloader = DataLoader(xray_test_dataset, batch_size=batch_size)

# model = MonaiResNet183DWrapper()
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1) 

model = nn.DataParallel(model)
model = model.cuda()
model = model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# cv_best_models = []
# cv_best_acc = []
# batch_size = 32
# epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# # Define dataset paths for training
# train_image_paths = []
# train_labels = []
# for label in [0, 1]:
#     folder = f"input_up_baseline/training/{label}"
#     for img_path in os.listdir(folder):
#         train_image_paths.append(os.path.join(folder, img_path))
#         train_labels.append(label)

# # Split the training dataset into training and validation sets
# split_idx = int(0.8 * len(train_image_paths))  # 80% for training, 20% for validation
# xray_train_image_paths = train_image_paths[:split_idx]
# xray_train_labels = train_labels[:split_idx]
# xray_val_image_paths = train_image_paths[split_idx:]
# xray_val_labels = train_labels[split_idx:]

# # Prepare the training dataset and dataloader
# xray_train_dataset = XrayDataset(xray_train_image_paths, xray_train_labels, transforms=transforms)
# xray_train_dataloader = DataLoader(xray_train_dataset, batch_size=batch_size, shuffle=True)

# # Prepare the validation dataset and dataloader
# xray_val_dataset = XrayDataset(xray_val_image_paths, xray_val_labels, transforms=transforms)
# xray_val_dataloader = DataLoader(xray_val_dataset, batch_size=batch_size)

best_val_acc = 0.0
best_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
best_model = None

for epoch in range(epochs):
    print(f"Epoch {epoch}")
    losses = []
    correct = 0

    for img, label in tqdm(xray_train_dataloader, leave=False):
        optimizer.zero_grad()
        
        out = model(img.cuda()).squeeze(1)
        loss = loss_fn(out, label.float().cuda())
        losses.append(loss.item())
        loss.backward()

        correct += torch.sum((label.cuda() == (out > 0.5))).item()
        
        optimizer.step()
    print(f"Training Loss: {np.mean(losses)}")
    print(f"Training Accuracy: {correct/(len(xray_train_dataset))}")



    correct = 0
    losses = 0
    all_preds = []
    all_labels = []
    for img, label in tqdm(xray_test_dataloader, leave=False):
        out = model(img.cuda()).squeeze(1)

        preds = (out > 0.5).float()
        loss = loss_fn(preds, label.float().cuda())

        losses += loss.item() * img.size(0)
        correct += torch.sum(preds == label.cuda())
        # Collect predictions and labels for metric computation
        all_preds.extend(preds.cpu().numpy())  
        all_labels.extend(label.cpu().numpy())


    final_acc = correct/len(xray_test_dataset)

    print(f"Validation Accuracy: {final_acc}")

    # Compute additional metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
    # Save the best model based on validation accuracy
    if final_acc > best_val_acc:
        best_val_acc = final_acc
        best_metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        best_model = copy.deepcopy(model)
        print("Best model updated!")

    scheduler.step()

# correct = 0 
# # losses = 0
# all_preds = []
# all_labels = []
# for img, label in tqdm(xray_test_dataloader, leave=False):
#     out = best_model(img.cuda()).squeeze(1)

#     preds = (out > 0.5).float()
#     loss = loss_fn(preds, label.float().cuda())

#     losses += loss.item() * img.size(0)
#     correct += torch.sum(preds == label.cuda())
#     # Collect predictions and labels for metric computation
#     all_preds.extend(preds.cpu().numpy())  
#     all_labels.extend(label.cpu().numpy())


# final_acc = correct/len(xray_test_dataset)

# print(f"Best Model Final Test Accuracy: {final_acc}")

# # Compute additional metrics
# accuracy = accuracy_score(all_labels, all_preds)
# precision = precision_score(all_labels, all_preds)
# recall = recall_score(all_labels, all_preds)
# f1 = f1_score(all_labels, all_preds)

# print(f"Accuracy: {accuracy:.6f}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
# # scheduler.step()