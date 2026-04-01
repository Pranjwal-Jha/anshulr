import torch
from model import SegCapsCNN
from preprocess import DeepfakePreprocessor
from blockchain_fl import BlockchainOrchestrator, ClientNode
import copy

def run_bfldl_workflow():
    """
    Orchestrates the entire BFLDL (Blockchain-based Federated Deep Learning) 
    workflow based on the paper.
    """
    print("Initializing BFLDL System...")

    # 1. System Setup (Blockchain Orchestrator)
    # The Global Model is stored and managed by the Blockchain
    global_model = SegCapsCNN()
    blockchain = BlockchainOrchestrator(global_model)
    
    # 2. Preprocessing Setup
    # Implements the normalization and sizing rules from the paper
    preprocessor = DeepfakePreprocessor(target_size=(299, 299))

    # 3. Client Setup (Federated Learning Nodes)
    # Simulate two local resources (e.g., hospitals or media agencies)
    # Each starts with the current global model state.
    client_ids = ["Resource_A", "Resource_B"]
    clients = [ClientNode(cid, copy.deepcopy(global_model)) for cid in client_ids]

    # 4. Data Loading Simulation
    # Paper uses 2D slices. We simulate this with dummy tensors for illustration.
    # In practice, you'd load images/video frames here.
    dummy_input = torch.randn(1, 3, 299, 299)
    # Class labels: [Real, Fake] capsules. [0, 1] means 'Fake'.
    # Paper mentions 97% accuracy for BFLDL on deepfake datasets.
    dummy_label = torch.tensor([[0.0, 1.0]])

    # 5. Federated Learning Loop (Block 1)
    print("\n--- Training Round 1 ---")
    for client in clients:
        # Step A: Local Training (on the client's private data)
        # Using the Capsule Network architecture (SegCapsCNN)
        client.train_locally(dummy_input, dummy_label, epochs=1)
        
        # Step B: Register Weights on the Blockchain
        # This is the 'Smart Contract' execution phase.
        blockchain.register_update(client.client_id, client.get_weights())

    # Step C: Consensus Mechanism (Blockchain Aggregation)
    # Aggregates local models (FedAvg) and updates the Global Model
    blockchain.aggregate_consensus()

    # Step D: Model Broadcast (Updating all clients)
    # The new Global Model is sent back to all participants
    new_global = blockchain.broadcast_model()
    for client in clients:
        client.sync_global_model(new_global)

    print("\n--- BFLDL Round 1 Completed Successfully ---")
    print("The decentralized global model is now more robust.")

if __name__ == "__main__":
    run_bfldl_workflow()
