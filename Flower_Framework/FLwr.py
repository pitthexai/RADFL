from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from flwr.common import Metrics, Context
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation
# from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from torchvision.models import resnet18, ResNet18_Weights
import os
import csv
import flwr as fl
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from flwr.simulation import start_simulation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

# Batch size and input image size constants
BATCH_SIZE = 32
IMAGE_SIZE = 224  

class XrayDataset(Dataset):
    """
    Custom Dataset for loading X-ray images from folder structure:
    root_dir/0 for negative class and root_dir/1 for positive class.
    """
    def __init__(self, root_dir, transforms=None):
        self.image_paths = []
        self.labels = []
        self.transforms = transforms

        print(f"Loading data from: {root_dir}")
        # Loop through '0' and '1' folders and collect image paths and labels
        for label in ['0', '1']:
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transforms:
            img = self.transforms(img)
        return {"img": img, "label": label}


def load_partitioned_datasets():
    """
    Load train, validation, and test datasets from local folder structure.
    Returns data loaders for training partitions and combined validation and test sets.
    """
    transform_ops = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        
    ])

    # Training datasets for clients B, C, D
    train_dataset_B = XrayDataset("/home/feg48/fl_xray_project/low_line/training/siteB", transforms=transform_ops)
    train_dataset_C = XrayDataset("/home/feg48/fl_xray_project/low_line/training/siteC", transforms=transform_ops)
    train_dataset_D = XrayDataset("/home/feg48/fl_xray_project/low_line/training/siteD", transforms=transform_ops)
    # Validation datasets
    val_dataset_B = XrayDataset("/home/feg48/fl_xray_project/low_line/validation/siteB", transforms=transform_ops)
    val_dataset_C = XrayDataset("/home/feg48/fl_xray_project/low_line/validation/siteC", transforms=transform_ops)
    val_dataset_D = XrayDataset("/home/feg48/fl_xray_project/low_line/validation/siteD", transforms=transform_ops)
    # Testing datasets
    test_dataset_B = XrayDataset("/home/feg48/fl_xray_project/low_line/testing/siteB", transforms=transform_ops)
    test_dataset_C = XrayDataset("/home/feg48/fl_xray_project/low_line/testing/siteC", transforms=transform_ops)
    test_dataset_D = XrayDataset("/home/feg48/fl_xray_project/low_line/testing/siteD", transforms=transform_ops)

    # DataLoaders for each training site
    train_loaders = {
        "B": DataLoader(train_dataset_B, batch_size=BATCH_SIZE, shuffle=True),
        "C": DataLoader(train_dataset_C, batch_size=BATCH_SIZE, shuffle=True),
        "D": DataLoader(train_dataset_D, batch_size=BATCH_SIZE, shuffle=True),
    }

    # Combined validation and test loaders
    valloader = DataLoader(
        ConcatDataset([val_dataset_B, val_dataset_C, val_dataset_D]),
        batch_size=BATCH_SIZE
    )

    testloader = DataLoader(
        ConcatDataset([test_dataset_B, test_dataset_C, test_dataset_D]),
        batch_size=BATCH_SIZE
    )
    
    # print("Evaluation dataset size:", len(valloader.dataset))

    return train_loaders, valloader, testloader

class Net(nn.Module):
    """
    Neural network model based on ResNet18 adapted for binary classification.
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x  

def get_parameters(net) -> List[np.ndarray]:
    """
    Extract model parameters as a list of numpy arrays for Flower.
    """
    print("Extracting model parameters...")
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """
    Set model parameters from a list of numpy arrays received from Flower.
    """
    print("Setting model parameters...")
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    """
    Train the model for a given number of epochs on the trainloader dataset.
    Records and prints loss and accuracy per epoch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Accumulate loss and accuracy metrics
            epoch_loss += loss.item() 
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        # Average loss over entire dataset
        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        # Append training metrics to CSV file
        csv_path = "client_each_train_metrics_prox.csv"
        write_header = not os.path.exists(csv_path)  # Write header only if file is new

        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if epoch == 0 and write_header:
                writer.writerow(["Epoch", "train_Loss", "train_Accuracy"])
            writer.writerow([epoch + 1, epoch_loss, epoch_acc])

    return {
        "train_loss": float(epoch_loss),
        "train_accuracy": float(epoch_acc),
    }

def test(net, test_loader):
    """
    Evaluate the model on the test/val dataset.
    Returns average loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)  # accumulate total loss

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss, accuracy, precision, recall, f1

# Connect the training in the pipeline using the Flower Client.
class FlowerClient(NumPyClient):
    """
    Flower client implementing train, evaluate and parameter exchange.
    """
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid  # partition ID of a client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        # Send model parameters to server
        print("Send model parameters to server")
        print(f"[Client {self.pid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Update local model and train
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        print("xxxxxxxxx")
        train_metrics = train(
            self.net,
            self.trainloader,
            epochs=local_epochs,
        )
        # train(self.net, self.trainloader, epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), train_metrics
    
    # client-side evaluation
    def evaluate(self, parameters, config):
        # Update local model and evaluate on validation set
        print(f"[Client {self.pid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        print("LLLLLLL")
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

# Client function
def client_fn(context: Context) -> Client:
    # Initialize a new model and move to device (CPU or GPU)
    net = Net()
    net.to(DEVICE)  
    # Mapping partition ID (0, 1, 2) to site names
    site_map = ["B", "C", "D"]
    partition_id = context.node_config["partition-id"]  # e.g. 0, 1, or 2
    site_id = site_map[partition_id]

    # Load datasets and select the one corresponding to this client/site
    trainloader, valloader, _ = load_partitioned_datasets()
    trainloader = trainloader[site_id]

    # Return a Flower client converted to the expected format
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()


# Create the ClientApp used in the simulation
client = ClientApp(client_fn=client_fn)

# Get initial model parameters to provide to the server at the start
params = get_parameters(Net())

# Global flag to check if file has been cleared
files_cleared = False

# Server-side evaluation function executed after each round
def evaluate( server_round: int, parameters: NDArrays, config: Dict[str, Scalar],) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = Net().to(DEVICE)

    global files_cleared
    csv_path = "/home/feg48/FL_FLwr/Server_side_eval_metrics_prox.csv"

    if not files_cleared:
        # Remove the file if exists, only once at start of training
        if os.path.exists(csv_path):
            os.remove(csv_path)
        files_cleared = True  # Append aggregated results to CSV

    # Load validation and test loaders
    _, valloader, testloader = load_partitioned_datasets()
    set_parameters(net, parameters)  # Set the model to current global parameters

    # Perform evaluation
    val_loss, val_acc, _, _, _ = test(net, valloader)
    print(f"[Round {server_round}] Validation — loss {val_loss} / accuracy {val_acc}")

    # Test evaluation
    _, test_acc, test_prec, test_recall, test_f1 = test(net, testloader)
    print(f"[Round {server_round}] Test —  Acc: {test_acc}, Prec: {test_prec}, Recall: {test_recall}, F1: {test_f1}")

    # Write results to CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Round", "val_Loss", "val_Acc", "test_Acc", "test_Prec", "test_Recall", "test_F1"])
        writer.writerow([server_round, val_loss, val_acc, test_acc, test_prec, test_recall, test_f1])

    return val_loss, {"val_accuracy": val_acc}

# Provide training configuration to clients each round
def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": 25,  # Placeholder: actual round is passed in `server_round` #30
        "local_epochs": 7    # Fixed for now; can be dynamic based on server_round #5
    }
    return config

# Global flag to check if file has been cleared
file_cleared = False

# Aggregates training metrics from clients after each round
def fit_metrics_aggregation_fn(metrics):
    global file_cleared
    csv_path = "server_train_metrics_prox.csv"

    if not file_cleared:
        # Remove the file if exists, only once at start of training
        if os.path.exists(csv_path):
            os.remove(csv_path)
        file_cleared = True

    total = sum(num_examples for num_examples, _ in metrics)
    loss = sum(num_examples * m["train_loss"] for num_examples, m in metrics) / total
    acc = sum(num_examples * m["train_accuracy"] for num_examples, m in metrics) / total
    print(f"[Server Aggregated Train] Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Append aggregated results to CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Train_Loss", "Train_Accuracy"])
        writer.writerow([loss, acc])
        
        
    return {"train_loss": loss, "train_accuracy": acc}

# Configuration passed to clients for evaluation
def evaluate_config(server_round):
    return {"round": server_round}

# Server-side function that defines strategy and training behavior
def server_fn(context: Context) -> ServerAppComponents:
    # # Create FedAvg strategy
#     strategy = FedAvg(
#         fraction_fit=1.0,  # 100% of available clients are selected for training in each round
#         fraction_evaluate=1.0,  # 100% of available clients are selected for evaluation in each round
#         min_fit_clients=3,  # Minimum number of clients required to participate in training
#         min_evaluate_clients=3,  # Minimum number of clients required to participate in evaluation
#         min_available_clients=3,  # Minimum number of total clients that must be connected to proceed with a round
#         initial_parameters=ndarrays_to_parameters(params),  # Initial global model parameters
#         evaluate_fn=evaluate,  # Custom evaluation function for server-side evaluation
#         on_fit_config_fn=fit_config,  # Function to configure training parameters (e.g., number of local epochs) for each round
#         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # <-- pass the metric aggregation function
#         on_evaluate_config_fn=evaluate_config,
#     )

    # Create FedProx strategy
    strategy = FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        proximal_mu=0.1,  # <-- FedProx regularization parameter (tune as needed)
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        on_evaluate_config_fn=evaluate_config,
    )
    config = ServerConfig(num_rounds= 30)    # Total number of federated learning rounds
    return ServerAppComponents(strategy=strategy, config=config)


# Create the server application
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
# backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}


# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=3,
    backend_config=backend_config,
)