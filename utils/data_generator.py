import numpy as np
import pandas as pd
from typing import Dict, Tuple

class TrafficDataGenerator:
    """
    Generates synthetic network traffic data for Federated Learning simulation.
    Features: Packet_Size, Inter_Arrival_Time, Protocol_Type, Bitrate.
    Classes: Streaming, VoIP, Gaming, Web_Browsing.
    """
    
    CLASSES = ['Streaming', 'VoIP', 'Gaming', 'Web_Browsing']
    FEATURES = ['Packet_Size', 'Inter_Arrival_Time', 'Protocol_Type', 'Bitrate']
    
    # Define feature characteristics for each class (Mean, Std Dev)
    # Protocol_Type: 0 for UDP, 1 for TCP (Probabilistic)
    CLASS_PROFILES = {
        'Streaming': {
            'Packet_Size': (1200, 200),       # Large packets
            'Inter_Arrival_Time': (5, 2),     # Constant stream, low jitter
            'Protocol_Type_TCP_Prob': 0.8,    # Mostly TCP (e.g. Netflix)
            'Bitrate': (5000, 1000)           # High bitrate
        },
        'VoIP': {
            'Packet_Size': (200, 50),         # Small packets
            'Inter_Arrival_Time': (20, 5),    # Regular small bursts
            'Protocol_Type_TCP_Prob': 0.1,    # Mostly UDP
            'Bitrate': (100, 20)              # Low bitrate
        },
        'Gaming': {
            'Packet_Size': (100, 30),         # Very small packets
            'Inter_Arrival_Time': (10, 8),    # Fast, but bursty
            'Protocol_Type_TCP_Prob': 0.5,    # Mix of UDP/TCP
            'Bitrate': (500, 150)             # Moderate bitrate
        },
        'Web_Browsing': {
            'Packet_Size': (800, 400),        # Variable
            'Inter_Arrival_Time': (500, 200), # Burst, then idle (High IAT)
            'Protocol_Type_TCP_Prob': 0.95,   # Mostly TCP (HTTP)
            'Bitrate': (1000, 800)            # Spiky
        }
    }

    @staticmethod
    def generate_client_data(num_samples: int, heterogeneity: str = "IID", client_id: int = 0, total_clients: int = 1) -> pd.DataFrame:
        """
        Generates data for a single client.
        
        Args:
            num_samples: Number of samples to generate.
            heterogeneity: "IID" (Uniform distribution) or "Non-IID" (Skewed).
            client_id: ID of the client (used for skewing Non-IID data).
            total_clients: Total number of clients to determine skew pattern.
            
        Returns:
            pd.DataFrame: Synthetic dataset with labels.
        """
        data = []
        labels = []
        
        # Determine class probabilities based on heterogeneity
        if heterogeneity == "IID":
            probs = [1/4] * 4
        else:
            # Non-IID: Skew each client towards a specific class preferentially
            # Simple circular shift logic: Client 0 -> Class 0 dominate, Client 1 -> Class 1 dominate...
            # This ensures different clients see different majorities
            primary_class_idx = client_id % 4
            probs = [0.1] * 4
            probs[primary_class_idx] = 0.7 # 70% of data comes from one class
        
        # Generate sample counts per class
        counts = np.random.multinomial(num_samples, probs)
        
        for i, class_name in enumerate(TrafficDataGenerator.CLASSES):
            n = counts[i]
            if n == 0: continue
            
            profile = TrafficDataGenerator.CLASS_PROFILES[class_name]
            
            # Generate features
            packet_sizes = np.random.normal(profile['Packet_Size'][0], profile['Packet_Size'][1], n)
            iats = np.random.normal(profile['Inter_Arrival_Time'][0], profile['Inter_Arrival_Time'][1], n)
            bitrates = np.random.normal(profile['Bitrate'][0], profile['Bitrate'][1], n)
            
            # Protocol Type: Binomial (0 or 1) based on probability of TCP
            protocols = np.random.binomial(1, profile['Protocol_Type_TCP_Prob'], n)
            
            # Clip values to realistic ranges (no negative sizes/times)
            packet_sizes = np.clip(packet_sizes, 40, 1500) # Min 40 bytes, Max 1500 regular MTU
            iats = np.clip(iats, 0, None)
            bitrates = np.clip(bitrates, 0, None)
            
            # Append features
            for j in range(n):
                data.append([packet_sizes[j], iats[j], protocols[j], bitrates[j]])
                labels.append(i) # Use numerical label (0-3)

        df = pd.DataFrame(data, columns=TrafficDataGenerator.FEATURES)
        df['Label'] = labels
        
        # Shuffle
        df = df.sample(frac=1, random_state=42 + client_id).reset_index(drop=True)
        
        return df

    @staticmethod
    def get_class_names() -> Dict[int, str]:
        return {i: name for i, name in enumerate(TrafficDataGenerator.CLASSES)}
