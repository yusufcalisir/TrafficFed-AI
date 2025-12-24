import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficClassifier(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for network traffic classification.
    Input: 4 features (Packet Size, IAT, Protocol, Bitrate)
    Output: 4 classes (Streaming, VoIP, Gaming, Web Browsing)
    """
    def __init__(self, input_dim=4, hidden_dim1=64, hidden_dim2=32, output_dim=4):
        super(TrafficClassifier, self).__init__()
        # Input Layer -> Hidden 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        # Hidden 1 -> Hidden 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # Hidden 2 -> Output
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
        # Dropout for regularization 
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Activation: ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (Logits)
        x = self.fc3(x)
        return x

def get_model():
    """Returns a fresh instance of the model."""
    return TrafficClassifier()
