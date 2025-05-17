import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from typing import List
import numpy as np
from tqdm import tqdm
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset, DataLoader
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Any, Dict, List
import argparse
import os
import copy
import torch

'''
python framework_fedprox.py --rounds 50 \
  --n_client_epochs 30\
  --batch_size 32 \
  --lr 0.005 \
  --log_every 2 \
  --img_size 224 \
  --model resnet18 \
  --mu 0.1 \
  --agg_method fedprox
'''

# Define custom dataset class
class XrayDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.image_paths = []
        self.labels = []
        self.transforms = transforms

        # Check root directory and subdirectories
        print(f"Loading data from: {root_dir}")

        for label in ['0', '1']:
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                # print(f"Found label directory: {label_path}")
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    # print(f"Found image: {img_path}")  # Debugging print
                    self.image_paths.append(img_path)
                    self.labels.append(int(label))                                      
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label

def prepare_transformer(args):

    image_size = args.img_size
    # Define transformations
    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return data_transforms

def prepare_dataloader(args, data_transforms):

    batch_size = args.batch_size

    train_dataset_B = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/training/siteB", transforms=data_transforms)
    train_dataset_C = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/training/siteC", transforms=data_transforms)
    train_dataset_D = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/training/siteD", transforms=data_transforms)

    val_dataset_B = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/testing/siteB", transforms=data_transforms)
    val_dataset_C = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/testing/siteC", transforms=data_transforms)
    val_dataset_D = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/testing/siteD", transforms=data_transforms)

    test_dataset_B = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/testing/siteB", transforms=data_transforms)
    test_dataset_C = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/testing/siteC", transforms=data_transforms)
    test_dataset_D = XrayDataset(root_dir="/home/feg48/fl_xray_project/low_line/testing/siteD", transforms=data_transforms)
        
    train_loader_B = DataLoader(train_dataset_B, batch_size=batch_size, shuffle=True)
    train_loader_C = DataLoader(train_dataset_C, batch_size=batch_size, shuffle=True)
    train_loader_D = DataLoader(train_dataset_D, batch_size=batch_size, shuffle=True)

    # Combine the datasets into a single dataset
    total_val_dataset = ConcatDataset([val_dataset_B, val_dataset_C, val_dataset_D])
    # Create a DataLoader for the combined dataset
    total_val_loader = DataLoader(total_val_dataset, batch_size=batch_size)

    # Combine the datasets into a single dataset
    total_test_dataset = ConcatDataset([test_dataset_B, test_dataset_C, test_dataset_D])
    # Create a DataLoader for the combined dataset
    total_test_loader = DataLoader(total_test_dataset, batch_size=batch_size)

    train_loaders = {
        "B": train_loader_B,
        "C": train_loader_C,
        "D": train_loader_D,
    }

    return train_loaders, total_val_loader, total_test_loader


def get_model_weights(model):
    n_par = 0
    for name, param in model.named_parameters():
        n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros(n_par).astype('float32')

    idx = 0
    for name, param in model.named_parameters():
        temp = param.data.cpu().numpy().reshape(-1)
        param_mat[idx:idx + len(temp)] = temp
        idx += len(temp)
    return np.copy(param_mat)

def average_weights(weights: List[Dict[str, torch.Tensor]], sample_size: List[int]) -> Dict[str, torch.Tensor]:
    # weights_avg = copy.deepcopy(weights[0])
    weights_avg = {key: weights[0][key] * (sample_size[0] / sum(sample_size)) for key in weights[0].keys()}


    n_samples = sum(sample_size)
    update_weights=([client_n_samples*1.0/n_samples for client_n_samples in sample_size])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key] * update_weights[i]

    return weights_avg


def _train_client(
    args, root_model, train_loader, client_id
) -> Tuple[nn.Module, float, float]:

    model = copy.deepcopy(root_model)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )
    root_model_wegiths = get_model_weights(root_model)

    for epoch in range(args.n_client_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for img, label in tqdm(train_loader, leave=False):
            img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()

            # logits = model(img)
            logits = F.log_softmax(model(img), dim=1)
            loss = F.nll_loss(logits, label)

            # FedProx regularization term
            local_model_weights = get_model_weights(model)
            prox_loss = (args.mu / 2) * np.sum((local_model_weights - root_model_wegiths) ** 2)
            loss += prox_loss  # Add regularization term
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(dim=1) == label).sum().item()
            epoch_samples += img.size(0)

            if torch.isnan(logits).any():
                print("Warning: NaN detected in logits")


        # Calculate average accuracy and loss
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = epoch_correct / epoch_samples


        print(f"Epoch {epoch + 1}/{args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}")

    # Final results after training all epochs
    print(f"\nClient #{client_id} | Training Complete")
    # print(f"  Final Loss: {epoch_loss} | Final Accuracy: {epoch_acc }\n")

    return model, epoch_loss, epoch_acc

def train(args, root_model, train_dataloaders, val_dataloaders, test_dataloader, model_folder) -> None:
    client_sites = ["B", "C", "D"]

    """Train a server model."""
    results = []

    for round in range(1, args.rounds + 1):
        clients_models = []
        clients_samples = []
        clients_accuracy = []
        clients_losses = []

        # # Randomly select m clients
        # m = random.choice([1,2,3])
        # selected_clients = random.sample(client_sites, m)
        # print(f"\n{round} round selected client: {selected_clients}")

        # Load client selection CSV
        client_selection_df = pd.read_csv("/home/feg48/fl_xray_project/fl_framework_models/selected_clients.csv")
        client_sites = list(client_selection_df.columns[1:])

        # Use min to avoid going out of bounds
        num_rounds = min(args.rounds, len(client_selection_df))

        for i in range(num_rounds):
            round_number = client_selection_df.iloc[i]['round']
            row = client_selection_df.iloc[i]
            selected_clients = [site for site in client_sites if row[site] == 1]

            # print(f"\nRound {round_number} selected clients: {selected_clients}")

        normalized_clients = [client_id.replace("site", "") for client_id in selected_clients]

        # Train clients
        root_model.train()

        for client_id in normalized_clients:
            train_loader = train_dataloaders[client_id]

            # Train client
            client_model, client_loss, client_acc = _train_client(
                args=args,
                root_model=root_model,
                train_loader=train_loader,
                client_id=client_id,
            )
            clients_models.append(client_model.state_dict())
            clients_losses.append(client_loss)
            clients_samples.append(len(train_loader.dataset))
            clients_accuracy.append(client_acc)

        updated_weights = average_weights(clients_models, clients_samples)
        
        # print(f"Before update: {list(root_model.parameters())[0][0][0:5]}")
        root_model.load_state_dict(updated_weights)
        # print(f"After update: {list(root_model.parameters())[0][0][0:5]}")

        # Update average loss of this round
        avg_client_loss = sum(clients_losses) / len(clients_losses)
        avg_client_acc = sum(clients_accuracy) / len(clients_accuracy)
        # print(f"Client {client_id} loss: {client_loss}")

        # Test server model on validation dataset
        val_loss, val_acc, val_precision, val_recall, val_f1 = test(root_model, val_dataloaders)

        # Test server model on testing dataset
        test_loss, test_acc, test_precision, test_recall, test_f1 = test(root_model, test_dataloader)

        # Print results to CLI
        print(f"\n\nResults after {round} rounds of training:")
        print(f"---> Training Loss: {avg_client_loss} | Training Accuracy: {avg_client_acc} | ")
        print(
            f"---> Val Loss: {val_loss} | Val Accuracy: {val_acc} | "
            f"Precision: {val_precision} | Recall: {val_recall} | F1-score: {val_f1}\n"
        )
        print(
            f"---> Test Loss: {test_loss} | Test Accuracy: {test_acc} | "
            f"Precision: {test_precision} | Recall: {test_recall} | F1-score: {test_f1}\n"
        )

        # save metrics for each rounds
        results.append((round, avg_client_loss, avg_client_acc, val_loss, val_acc, val_precision, val_recall, val_f1, test_loss, test_acc, test_precision, test_recall, test_f1))

        if round % args.log_every == 0:
            model_file_path = os.path.join(model_folder, 'round_{round}.pth'.format(round=round))
            
            best_model = copy.deepcopy(root_model)
            best_model = root_model.state_dict()  # Save the model's state_dict
            torch.save(best_model, model_file_path)

            # save csv
            df = pd.DataFrame(results, columns=["round", "train_lose", "train_acc", "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "test_loss", "test_acc", "test_precision", "test_recall", "test_f1"])
            result_file_path = os.path.join(model_folder, 'result_{round}.csv'.format(round=round))
            df.to_csv(result_file_path, index=False)
    
    # save the last model
    model_file_path = os.path.join(model_folder, 'last.pth')
    best_model = copy.deepcopy(root_model)
    best_model = root_model.state_dict()  # Save the model's state_dict
    torch.save(best_model, model_file_path)

     # save result metrics
    df = pd.DataFrame(results, columns=["round", "train_lose", "train_acc", "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "test_loss", "test_acc", "test_precision", "test_recall", "test_f1"])
    result_file_path = os.path.join(model_folder, 'final_results.csv')
    df.to_csv(result_file_path, index=False)

def test(root_model, test_dataloader) -> Tuple[float, float, float, float, float]:
    """Test the server model."""

    root_model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    for img, label in tqdm(test_dataloader):
        img, label = img.cuda(), label.cuda()

        # logits = root_model(img)
        logits = F.log_softmax(root_model(img), dim=1)
        loss = F.nll_loss(logits, label)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == label).sum().item()
        total_samples += img.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    # calculate average accuracy and loss
    total_loss /= len(test_dataloader.dataset)
    total_acc = total_correct / len(test_dataloader.dataset)

    # print(f"Debug: Final Test Loss: {total_loss}, Final Test Accuracy: {total_acc}")
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return total_loss, total_acc, precision, recall, f1



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--n_client_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument("--agg_method", type=str, default="fedprox", choices=['fedavg', 'fedprox', 'feddyn', 'fedgen', 'fedsam', 'fedscaffold'], help='choose method')

    args = parser.parse_args()

    print(args)
    
    data_transforms = prepare_transformer(args)

    train_dataloaders, val_dataloaders, test_dataloader = prepare_dataloader(args, data_transforms)

    # define resnet18 model
    # if args.model == 'resnet18': TODO: use if-condition to change model
    root_model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = root_model.fc.in_features
    root_model.fc = nn.Linear(num_ftrs, 2)
    root_model = nn.DataParallel(root_model)
    root_model = root_model.cuda()

    # Define path to save the model
    model_name = '{}_{}_rounds_{}_nclientepochs_{}_lr_{}_batchsize_{}_imgsize_{}_mu_{}_trial_{}'.format(args.agg_method, args.model, args.rounds, args.n_client_epochs, args.lr, args.batch_size, args.img_size, args.mu, args.trial)
    model_folder = os.path.join("/home/feg48/fl_xray_project/model/", model_name)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    
    # Start training
    train(args, root_model, train_dataloaders, val_dataloaders, test_dataloader, model_folder)


if __name__ == "__main__":
    main()