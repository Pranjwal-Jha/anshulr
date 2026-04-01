import torch
import copy
from collections import OrderedDict
from model import SegCapsCNN

class BlockchainOrchestrator:
    """
    Simulates the Blockchain layer in BFLDL. 
    It manages the Global Model, collects local weights, 
    and performs the consensus-based aggregation.
    """
    def __init__(self, initial_model):
        self.global_model = initial_model
        self.local_updates = []
        self.ledger = [] # To store hashes/records of updates

    def register_update(self, client_id, model_weights):
        """
        Equivalent to a blockchain transaction: 
        Clients 'commit' their locally trained weights.
        """
        # In a real blockchain, this would involve hashing, signing,
        # and validating with PoW/PoS consensus.
        update = {
            'client_id': client_id,
            'weights': copy.deepcopy(model_weights)
        }
        self.local_updates.append(update)
        print(f"Update from Client {client_id} registered on Blockchain.")

    def aggregate_consensus(self):
        """
        BFLDL consensus: Aggregates weights using Federated Averaging (FedAvg).
        This becomes the 'Global Model' for the next block.
        """
        if not self.local_updates:
            return

        # Simple Federated Averaging (FedAvg)
        avg_weights = OrderedDict()
        num_updates = len(self.local_updates)
        
        for i, update in enumerate(self.local_updates):
            weights = update['weights']
            for key in weights.keys():
                if i == 0:
                    avg_weights[key] = weights[key] / num_updates
                else:
                    avg_weights[key] += weights[key] / num_updates

        # Update the Global Model
        self.global_model.load_state_dict(avg_weights)
        
        # Clear local updates for next round
        self.local_updates = []
        # 'Commit' the block
        self.ledger.append(f"Global model updated by {num_updates} clients.")
        print("Blockchain consensus achieved. Global Model updated.")

    def broadcast_model(self):
        """
        Broadcast the updated global model back to all resources/clients.
        """
        return copy.deepcopy(self.global_model)

class ClientNode:
    """
    Represents a data source (Client 1, Client 2, etc.) in the BFLDL scheme.
    Each node trains locally on its private deepfake data.
    """
    def __init__(self, client_id, local_model):
        self.client_id = client_id
        self.model = local_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train_locally(self, dummy_data, dummy_labels, epochs=1):
        """
        Local training loop using the Capsule Network (SegCapsCNN).
        """
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            # Predict
            outputs = self.model(dummy_data)
            # Binary cross entropy: [Real, Fake]
            loss = torch.nn.functional.mse_loss(outputs, dummy_labels)
            loss.backward()
            self.optimizer.step()
            print(f"Client {self.client_id} - Epoch {epoch+1} Local Loss: {loss.item():.4f}")

    def get_weights(self):
        return self.model.state_dict()

    def sync_global_model(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

if __name__ == "__main__":
    # Simulation logic
    print("--- BFLDL Simulation Start ---")
    
    # Initialize the Blockchain with a Global Model
    global_model = SegCapsCNN()
    blockchain = BlockchainOrchestrator(global_model)
    
    # Create two clients (Client 1 and Client 2)
    client1 = ClientNode("1", copy.deepcopy(global_model))
    client2 = ClientNode("2", copy.deepcopy(global_model))
    
    # Simulation Round 1
    # Mock some data: Batch of 1 image 299x299, 
    # Labels: 1.0 for "Fake" capsule, 0.0 for "Real"
    mock_data = torch.randn(1, 3, 299, 299)
    mock_labels = torch.tensor([[0.0, 1.0]]) # Class 1 (Fake)
    
    # Local Training
    client1.train_locally(mock_data, mock_labels, epochs=2)
    client2.train_locally(mock_data, mock_labels, epochs=2)
    
    # Send Weights to Blockchain
    blockchain.register_update(client1.client_id, client1.get_weights())
    blockchain.register_update(client2.client_id, client2.get_weights())
    
    # Blockchain Consensus (FedAvg)
    blockchain.aggregate_consensus()
    
    # Broadcast and Sync
    new_global_model = blockchain.broadcast_model()
    client1.sync_global_model(new_global_model)
    client2.sync_global_model(new_global_model)
    
    print("--- BFLDL Simulation Round 1 Completed ---")
