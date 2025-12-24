import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader, TensorDataset
import time

from utils.data_generator import TrafficDataGenerator
from utils.model import get_model, TrafficClassifier
from utils.federated import FederatedClient, FederatedServer

# --- Page Config ---
st.set_page_config(page_title="TrafficFed-AI", layout="wide", page_icon="🌐")

st.title("🌐 TrafficFed-AI: Federated Learning for Network Traffic Classification")
st.markdown("""
This dashboard simulates a **Federated Learning** system where multiple network devices (Clients) 
collaboratively train a generic Neural Network to classify traffic types (**Streaming, VoIP, Gaming, Web**) 
without sharing their raw data.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
num_clients = st.sidebar.slider("Number of Clients", 2, 10, 3)
num_rounds = st.sidebar.slider("Communication Rounds (Epochs)", 1, 20, 5)
local_epochs = st.sidebar.slider("Local Epochs per Round", 1, 5, 2)
heterogeneity = st.sidebar.radio("Data Distribution", ["IID", "Non-IID"])
noise_level = st.sidebar.slider("Diff. Privacy Noise (Sigma)", 0.0, 0.1, 0.0, step=0.01)

# --- Session State Management ---
if 'clients' not in st.session_state:
    st.session_state.clients = []
if 'global_model' not in st.session_state:
    st.session_state.global_model = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'history' not in st.session_state:
    st.session_state.history = {'loss': [], 'accuracy': []}
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

# --- Step 1: Data Generation ---
st.header("1. Initialize Network & Data")

if st.button("Generate Distributed Data"):
    st.session_state.clients = []
    st.session_state.history = {'loss': [], 'accuracy': []}
    st.session_state.training_complete = False
    
    # 1. Generate Global Test Set (Validates the Global Model)
    test_df = TrafficDataGenerator.generate_client_data(num_samples=500, heterogeneity="IID")
    st.session_state.test_data = test_df
    
    # 2. Generate Client Data
    client_data_summary = []
    
    for i in range(num_clients):
        # Generate data
        df = TrafficDataGenerator.generate_client_data(
            num_samples=300, 
            heterogeneity=heterogeneity, 
            client_id=i,
            total_clients=num_clients
        )
        
        # Instantiate FederatedClient and store
        client = FederatedClient(client_id=i, df_data=df, model_cls=TrafficClassifier)
        st.session_state.clients.append(client)
        
        # Collect stats for visualization
        counts = df['Label'].value_counts().sort_index()
        for label, count in counts.items():
            class_name = TrafficDataGenerator.CLASSES[label]
            client_data_summary.append({'Client': f'Client {i+1}', 'Class': class_name, 'Count': count})
            
    # Initialize Global Model
    st.session_state.global_model = get_model()
    
    st.success(f"Generated data for {num_clients} clients and initialized Global Model.")
    
    # Visualization: Class Distribution
    if client_data_summary:
        df_viz = pd.DataFrame(client_data_summary)
        fig = px.bar(df_viz, x="Client", y="Count", color="Class", 
                     title="Class Distribution per Client (Check for Heterogeneity)",
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)

# --- Step 2: Training Loop ---
if st.session_state.global_model:
    st.header("2. Federated Training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        start_btn = st.button("Start Federated Training", disabled=st.session_state.training_complete)
        
    if start_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare Plot
        chart_loss = st.empty()
        chart_acc = st.empty()
        
        # Test Loader
        X_test = st.session_state.test_data.drop('Label', axis=1).values.astype('float32')
        y_test = st.session_state.test_data['Label'].values.astype('int64')
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        for round_idx in range(num_rounds):
            status_text.text(f"Round {round_idx+1}/{num_rounds}: Clients training locally...")
            
            global_weights = st.session_state.global_model.state_dict()
            client_weights = []
            
            # 1. Local Training
            for client in st.session_state.clients:
                w = client.train(global_weights, epochs=local_epochs, noise_level=noise_level)
                client_weights.append(w)
                
            # 2. Server Aggregation
            status_text.text(f"Round {round_idx+1}/{num_rounds}: Server aggregating weights...")
            FederatedServer.aggregate(st.session_state.global_model, client_weights)
            
            # 3. Evaluation
            val_loss, val_acc = FederatedServer.evaluate(st.session_state.global_model, test_loader)
            st.session_state.history['loss'].append(val_loss)
            st.session_state.history['accuracy'].append(val_acc)
            
            # Update Charts
            
            # Loss Chart
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=st.session_state.history['loss'], mode='lines+markers', name='Loss'))
            fig_loss.update_layout(title="Global Model Loss", xaxis_title="Round", yaxis_title="Loss")
            chart_loss.plotly_chart(fig_loss, use_container_width=True)
            
            # Accuracy Chart
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=st.session_state.history['accuracy'], mode='lines+markers', name='Accuracy', line=dict(color='green')))
            fig_acc.update_layout(title="Global Model Accuracy (%)", xaxis_title="Round", yaxis_title="Accuracy")
            chart_acc.plotly_chart(fig_acc, use_container_width=True)
            
            progress_bar.progress((round_idx + 1) / num_rounds)
            time.sleep(0.5) # For demo effect
            
        st.session_state.training_complete = True
        status_text.success("Federated Training Complete!")

# --- Step 3: Inference ---
if st.session_state.training_complete:
    st.header("3. Real-time Inference")
    st.markdown("Test the trained Global Model with manual inputs.")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p_size = st.number_input("Packet Size (Bytes)", 0, 2000, 1200)
    with c2:
        iat = st.number_input("Inter-Arrival Time (ms)", 0.0, 100.0, 5.0)
    with c3:
        proto = st.selectbox("Protocol", ["UDP", "TCP"])
        proto_val = 1 if proto == "TCP" else 0
    with c4:
        bitrate = st.number_input("Bitrate (kbps)", 0, 10000, 5000)
        
    if st.button("Classify Value"):
        # Predict
        input_tensor = torch.tensor([[float(p_size), float(iat), float(proto_val), float(bitrate)]])
        
        model = st.session_state.global_model
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(probs).item()
            pred_class = TrafficDataGenerator.CLASSES[pred_idx]
            confidence = probs[0][pred_idx].item() * 100
            
        st.metric(label="Predicted Traffic Type", value=pred_class, delta=f"{confidence:.2f}% Confidence")
        st.bar_chart(pd.DataFrame({'Probability': probs[0].numpy()}, index=TrafficDataGenerator.CLASSES))

elif not st.session_state.clients:
    st.info("Please generate data first.")
