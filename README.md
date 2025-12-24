
# 🌐 TrafficFed-AI: Privacy-Preserving Network Traffic Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

**TrafficFed-AI** is a simulation framework demonstrating how **Federated Learning (FL)** can be used to classify encrypted network traffic (e.g., Streaming vs. Gaming) without violating user privacy.

Unlike traditional Deep Packet Inspection (DPI) methods that inspect packet content, this project uses **packet metadata** (size, timing, bitrate) to train a global model collaboratively across multiple clients, keeping raw data local and secure.

---

## 🚀 Key Features

* **Federated Averaging (FedAvg):** Simulates a central server aggregating weights from multiple distributed clients (IoT devices/Routers).
* **Encrypted Traffic Analysis:** Classifies traffic based on statistical flow features (Packet Size, Inter-Arrival Time, Protocol) without decryption.
* **Privacy-First:** Raw user data never leaves the client. Implements **Differential Privacy** by injecting noise into model updates.
* **Interactive Dashboard:** Built with **Streamlit** for real-time visualization of the training process, loss curves, and class distribution.
* **Heterogeneous Data Simulation:** Supports Non-IID data generation (e.g., Client A only does Gaming, Client B only watches Netflix) to simulate real-world scenarios.

---

## 🧠 How It Works

### The Problem
ISPs need to classify traffic to manage Quality of Service (QoS). However, modern traffic is encrypted (HTTPS/TLS), and inspecting packet payloads violates user privacy laws (GDPR/KVKK).

### The Solution
TrafficFed-AI trains a Neural Network on **metadata fingerprints**:

1.  **Local Training:** Each client (home router) trains a local model on its own traffic logs.
2.  **Model Updates:** Clients send only the *model weights* (mathematical parameters) to the server, not the data.
3.  **Aggregation:** The server averages these weights to create a smarter "Global Model."
4.  **Deployment:** The Global Model is sent back to all clients to classify new traffic.

---

## 📊 Traffic Classes & Features

The model classifies traffic into four distinct categories:

| Traffic Class | Characteristics | Example Applications |
| :--- | :--- | :--- |
| **Streaming** | Large packets, high bitrate, buffered bursts | Netflix, YouTube, Twitch |
| **VoIP** | Small packets, very frequent, UDP protocol | Zoom, Skype, WhatsApp Calls |
| **Gaming** | Small packets, high frequency, strict latency | CS:GO, Valorant, LoL |
| **Web Browsing**| Burst/Idle behavior, variable packet sizes | Chrome, Social Media feeds |

**Input Features used for prediction:**
* `Packet Size` (Bytes)
* `Inter-Arrival Time` (ms)
* `Bitrate` (kbps)
* `Protocol` (TCP/UDP)

---

## 🛠️ Tech Stack

* **Core Logic:** Python 3.x
* **Machine Learning:** PyTorch (Neural Networks)
* **UI / Dashboard:** Streamlit
* **Visualization:** Plotly (Interactive Charts)
* **Data Processing:** Pandas, NumPy, Scikit-learn

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/TrafficFed-AI.git](https://github.com/yusufcalisir/TrafficFed-AI.git)
cd TrafficFed-AI

```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio streamlit pandas numpy plotly scikit-learn

```

### 4. Run the Application

```bash
streamlit run app.py

```

*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 📂 Project Structure

```bash
TrafficFed-AI/
├── app.py                 # Main Streamlit application entry point
├── requirements.txt       # List of python dependencies
└── utils/
    ├── __init__.py
    ├── data_generator.py  # Synthetic traffic data generation logic
    ├── model.py           # PyTorch Neural Network architecture
    └── federated.py       # Client/Server classes and FedAvg algorithm

```

---

## 🔮 Future Improvements

* Integration with real packet capture tools (Wireshark/Scapy) for live traffic analysis.
* Implementation of advanced aggregation algorithms (FedProx) for better Non-IID performance.
* Secure Multi-Party Computation (SMPC) for enhanced security.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

**Developed by Yusuf Calisir**
*Researching the intersection of Privacy-Preserving AI and Network Security.*

```

