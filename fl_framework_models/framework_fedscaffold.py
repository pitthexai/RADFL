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
python framework_fedscaffold.py --rounds 1 \
  --n_client_epochs 3\
  --batch_size 32 \
  --local_lr 0.005 \
  --global_lr 1\
  --log_every 2 \
  --img_size 224 \
  --model resnet18 \
  --agg_method fedscaffold
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

    train_dataset_B = XrayDataset(root_dir="/home/feg48/fl_xray_project/input_low_baseline/training/siteB", transforms=data_transforms)
    train_dataset_C = XrayDataset(root_dir="/home/feg48/fl_xray_project/input_low_baseline/training/siteC", transforms=data_transforms)
    train_dataset_D = XrayDataset(root_dir="/home/feg48/fl_xray_project/input_low_baseline/training/siteD", transforms=data_transforms)

    test_dataset_B = XrayDataset(root_dir="/home/feg48/fl_xray_project/input_low_baseline/testing/siteB", transforms=data_transforms)
    test_dataset_C = XrayDataset(root_dir="/home/feg48/fl_xray_project/input_low_baseline/testing/siteC", transforms=data_transforms)
    test_dataset_D = XrayDataset(root_dir="/home/feg48/fl_xray_project/input_low_baseline/testing/siteD", transforms=data_transforms)
        
    train_loader_B = DataLoader(train_dataset_B, batch_size=batch_size, shuffle=True)
    train_loader_C = DataLoader(train_dataset_C, batch_size=batch_size, shuffle=True)
    train_loader_D = DataLoader(train_dataset_D, batch_size=batch_size, shuffle=True)

    # Combine the datasets into a single dataset
    total_test_dataset = ConcatDataset([test_dataset_B, test_dataset_C, test_dataset_D])
    # Create a DataLoader for the combined dataset
    total_test_loader = DataLoader(total_test_dataset, batch_size=batch_size)

    train_loaders = {
        "B": train_loader_B,
        "C": train_loader_C,
        "D": train_loader_D,
    }

    return train_loaders, None, total_test_loader

def update_model_from_weights(model, weights):
    # Access the actual model inside DataParallel
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model

    idx = 0
    for name, param in model_module.named_parameters():
        param_length = param.numel()
        new_values = weights[idx:idx + param_length]
        new_tensor = torch.tensor(new_values, dtype=torch.float32).to(param.device).view_as(param)

        # Update param in-place
        param.data.copy_(new_tensor)
        idx += param_length
    
    assert idx == len(weights), f"Mismatch in number of parameters: used {idx}, but got {len(weights)}"
    return model


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

def get_model_gradients(model):
    n_par = 0
    for name, param in model.named_parameters():
        n_par += len(param.grad.reshape(-1))
    
    grad_mat = np.zeros(n_par).astype('float32')

    idx = 0
    for name, param in model.named_parameters():
        temp = param.grad.detach().cpu().numpy().reshape(-1)
        grad_mat[idx:idx + len(temp)] = temp
        idx += len(temp)
    
    return np.copy(grad_mat)

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
    args, root_model, train_loader, client_id, c_avg, c_client
) -> Tuple[nn.Module, float, float, np.ndarray]:

    model = copy.deepcopy(root_model)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.local_lr, momentum=args.momentum
    )
    root_model_weights = get_model_weights(root_model)

    step_count = 0
    for epoch in range(args.n_client_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for img, label in tqdm(train_loader, leave=False):
            img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()

            # logits = F.log_softmax(model(img), dim=1)
            logits = model(img)
            logits = torch.clamp(logits, min=-10, max=10)  # avoid exploding values
            logits = F.log_softmax(logits, dim=1)

            loss = F.nll_loss(logits, label)

            # Optionally clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()

            model_gradients = get_model_gradients(model)

            # Apply control variate update: grad_batch - c_i + c
            adjusted_grads = model_gradients - c_client + c_avg

            # Normalize gradients if norm > 1.0
            grad_norm = np.linalg.norm(adjusted_grads)
            if grad_norm > 1.0:
                adjusted_grads = adjusted_grads / grad_norm

            # SGD update: w = w - lr * adjusted_grad
            model_weights = get_model_weights(model)
            model_weights = model_weights - args.local_lr * adjusted_grads
            model = update_model_from_weights(model, model_weights)

            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(dim=1) == label).sum().item()
            epoch_samples += img.size(0)
            step_count += 1

            if torch.isnan(logits).any():
                print("Warning: NaN detected in logits")


        # Calculate average accuracy and loss
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = epoch_correct / epoch_samples


        print(f"Epoch {epoch + 1}/{args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}")

    # Final results after training all epochs
    print(f"\nClient #{client_id} | Training Complete")
    # print(f"  Final Loss: {epoch_loss} | Final Accuracy: {epoch_acc }\n")

    # Save old c_client before updating it
    c_old = c_client.copy()

    # Update control variate c_i
    local_model_weights = get_model_weights(model)
    K = step_count
    rho = args.momentum
    lr = args.local_lr

    # Compute divisor depending on adaptive_divison flag
    if args.adaptive_divison:
        new_K = (K - rho * (1.0 - pow(rho, K)) / (1.0 - rho)) / (1.0 - rho)
        divisor = 1.0 / (new_K * lr)
    else:
        divisor = 1.0 / (K * lr)

    param_move = root_model_weights - local_model_weights
    # Compute the new c_client (which is c_new)
    c_new = c_client - c_avg + divisor * param_move

    # Compute delta_c as required by SCAFFOLD
    delta_c = c_new - c_old

    return model, epoch_loss, epoch_acc, delta_c

def train(args, root_model, train_dataloaders, val_dataloaders, test_dataloader, model_folder) -> None:
    client_sites = ["B", "C", "D"]
    K = len(client_sites)
    # # Initialize server state h
    # n_par = 0
    # for name, param in root_model.named_parameters():
    #     n_par += len(param.data.reshape(-1))

    # Initialize global control variate c
    n_par = sum(p.numel() for p in root_model.parameters())
    c_avg  = np.zeros(n_par).astype('float32')

    """Train a server model."""
    results = []

    # initial t-1 gradients as zeros
    c_clients = {
        "B": np.zeros(n_par).astype('float32'), 
        "C": np.zeros(n_par).astype('float32'), 
        "D": np.zeros(n_par).astype('float32'),
    }

    for round in range(1, args.rounds + 1):
        print(f"\n--- Round {round} ---")
        local_weights = []
        local_sizes = []
        local_losses = []
        local_accuracies = []
        c_updates = []
        

        # Randomly select m clients
        m = random.choice([1,2,3])
        selected_clients = random.sample(client_sites, m)
        print(f"\n{round} round selected client: {selected_clients}")

        # Train clients
        root_model.train()

        for client_id in selected_clients:
            print(f"\nTraining client: {client_id}")
            train_loader = train_dataloaders[client_id]

            # Train client
            client_model, loss, acc, delta_c = _train_client(
                args=args,
                root_model=root_model,
                train_loader=train_loader,
                client_id=client_id,
                c_avg=c_avg,
                c_client=c_clients[client_id]
            )
            weights = get_model_weights(client_model)
            local_weights.append(weights)
            local_sizes.append(len(train_loader.dataset))
            local_losses.append(loss)
            local_accuracies.append(acc)
            c_updates.append(delta_c)

            # Update client control variate
            c_clients[client_id] += delta_c


        # Aggregate client model updates (weighted average)
        total_samples = sum(local_sizes)
        local_weights = np.stack(local_weights, axis=0)
        weighted_avg_weights = sum(w * s for w, s in zip(local_weights, local_sizes)) / total_samples

        # Update global model
        current_weights = get_model_weights(root_model)
        updated_weights = current_weights + args.global_lr * (weighted_avg_weights - current_weights)

        # Update global control variate c_avg
        stacked_deltas = np.stack(c_updates)
        c_avg += (len(selected_clients) / K) * np.mean(stacked_deltas, axis=0)

        print(f"Before update: {list(root_model.parameters())[0][0][0:5]}")
        root_model = update_model_from_weights(root_model, updated_weights)
        print(f"After update: {list(root_model.parameters())[0][0][0:5]}")

        # Evaluate
        avg_loss = np.mean(local_losses)
        avg_acc = np.mean(local_accuracies)
        test_loss, test_acc, precision, recall, f1 = test(root_model, test_dataloader)

        # Test server model
        test_loss, test_acc, precision, recall, f1 = test(root_model, test_dataloader)

        # Print results to CLI
        print(f"\n\nResults after {round} rounds of training:")
        print(f"---> Training Loss: {avg_loss} | Training Accuracy: {avg_acc} | ")
        print(
            f"---> Test Loss: {test_loss} | Test Accuracy: {test_acc} | "
            f"Precision: {precision} | Recall: {recall} | F1-score: {f1}\n"
        )

        # save metrics for each rounds
        results.append((round, avg_loss, avg_acc, test_loss, test_acc, precision, recall, f1))

        if round % args.log_every == 0:
            model_file_path = os.path.join(model_folder, 'round_{round}.pth'.format(round=round))
            
            best_model = copy.deepcopy(root_model)
            best_model = root_model.state_dict()  # Save the model's state_dict
            torch.save(best_model, model_file_path)
    
    # save the last model
    model_file_path = os.path.join(model_folder, 'last.pth')
    best_model = copy.deepcopy(root_model)
    best_model = root_model.state_dict()  # Save the model's state_dict
    torch.save(best_model, model_file_path)

    # save result metrics
    df = pd.DataFrame(results, columns=["round", "train_lose", "train_acc", "test_loss", "test_acc", "test_precision", "test_recall", "test_f1"])
    result_file_path = os.path.join(model_folder, 'results.csv')
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
    parser.add_argument("--local_lr", type=float, default=0.001)
    parser.add_argument("--global_lr", type=float, default=1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument("--agg_method", type=str, default="fedscaffold", help='choose method')

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
    model_name = '{}_{}_rounds_{}_nclientepochs_{}_global_lr_{}_local_lr_{}_batchsize_{}_imgsize_{}_trial_{}'.format(args.agg_method, args.model, args.rounds, args.n_client_epochs, args.global_lr, args.local_lr, args.batch_size, args.img_size, args.trial)
    model_folder = os.path.join("/home/feg48/fl_xray_project/model/", model_name)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    
    # Start training
    train(args, root_model, train_dataloaders, val_dataloaders, test_dataloader, model_folder)


if __name__ == "__main__":
    main()