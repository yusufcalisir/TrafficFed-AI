import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from typing import List, Dict

class FederatedClient:
    """
    Simulates a client in the Federated Learning system.
    Holds local data and performs local model training.
    """
    def __init__(self, client_id: int, df_data, model_cls):
        self.client_id = client_id
        self.data = df_data
        self.model_cls = model_cls
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare Data
        X = self.data.drop('Label', axis=1).values.astype('float32')
        y = self.data['Label'].values.astype('int64')
        
        # Create TensorDataset
        self.dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        
    def train(self, global_model_state_dict: Dict, epochs: int = 5, lr: float = 0.01, noise_level: float = 0.0):
        """
        Trains the local model using the global model as a starting point.
        Args:
            global_model_state_dict: Weights from the server.
            epochs: Number of local training epochs.
            lr: Learning rate.
            noise_level: Standard deviation of Gaussian noise for Differential Privacy.
        Returns:
            dict: The state_dict of the trained local model.
        """
        # Initialize local model with global weights
        local_model = self.model_cls().to(self.device)
        local_model.load_state_dict(global_model_state_dict)
        local_model.train()
        
        optimizer = optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        
        # Local Training Loop
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        # Simulate Differential Privacy (Add Gaussian Noise)
        if noise_level > 0.0:
            with torch.no_grad():
                for param in local_model.parameters():
                    noise = torch.normal(mean=0.0, std=noise_level, size=param.size()).to(self.device)
                    param.add_(noise)
                    
        return local_model.state_dict()

class FederatedServer:
    """
    Simulates the central server in Federated Learning.
    Aggregates weights from clients (FedAvg).
    """
    @staticmethod
    def aggregate(global_model, client_weights: List[Dict]):
        """
        FedAvg Algorithm: Averages the weights from all clients.
        Args:
            global_model: The current global model (will be updated).
            client_weights: List of state_dicts from clients.
        """
        # Initialize an empty dictionary to store averaged weights
        avg_weights = copy.deepcopy(client_weights[0])
        
        # Sum up weights
        for key in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
                
            # Divide by number of clients to get average
            avg_weights[key] = torch.div(avg_weights[key], len(client_weights))
            
        # Update Global Model
        global_model.load_state_dict(avg_weights)
        
    @staticmethod
    def evaluate(model, test_loader, device=None):
        """
        Evaluates the global model on a test set.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                running_loss += loss.item() * data.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        avg_loss = running_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
