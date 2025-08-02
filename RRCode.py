import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import random

# Configuration
class Config:
    def __init__(self):
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.epochs_siamese = {1: 30, 2: 20, 3: 10, 4: 5}
        self.epochs_triplet = {1: 25, 2: 15, 3: 7, 4: 4}
        self.alpha = 0.4  # For target similarity calculation
        self.margin = 1.0  # For triplet loss
        self.embedding_dim = 768  # BERT embedding dimension
        self.max_seq_length = 128
        self.top_k = [1, 3, 5]  # For evaluation metrics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentence-BERT based document encoder
class DocumentEncoder:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
    
    def get_embedding_dim(self):
        return self.model.get_sentence_embedding_dimension()

# Dataset classes
class SiameseDataset(Dataset):
    def __init__(self, queries, positives, negatives, citation_graph, alpha=0.4):
        self.pairs = []
        self.labels = []
        
        # Create positive pairs (query, positive, 1)
        for q_idx, q_text in enumerate(queries):
            for p_idx in positives[q_idx]:
                path_length = nx.shortest_path_length(citation_graph, source=q_idx, target=p_idx)
                target_sim = alpha ** (path_length - 1)
                self.pairs.append((q_text, queries[p_idx], target_sim))
                self.labels.append(1)
        
        # Create negative pairs (query, negative, 0)
        for q_idx, q_text in enumerate(queries):
            for n_idx in negatives[q_idx]:
                self.pairs.append((q_text, queries[n_idx], 0))
                self.labels.append(0)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

class TripletDataset(Dataset):
    def __init__(self, queries, positives, negatives, citation_graph):
        self.triplets = []
        
        # Create triplets (query, positive, negative)
        for q_idx, q_text in enumerate(queries):
            for p_idx in positives[q_idx]:
                for n_idx in negatives[q_idx]:
                    self.triplets.append((q_text, queries[p_idx], queries[n_idx]))
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]

# Network architectures
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.encoder = DocumentEncoder()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward_once(self, x):
        return self.encoder.encode([x])[0] if isinstance(x, str) else self.encoder.encode(x)
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # Additional features for similarity prediction
        combined = torch.cat((output1, output2, torch.abs(output1 - output2)), dim=-1)
        similarity = self.fc(combined)
        return similarity.squeeze()

class TripletNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(TripletNetwork, self).__init__()
        self.encoder = DocumentEncoder()
    
    def forward_once(self, x):
        return self.encoder.encode([x])[0] if isinstance(x, str) else self.encoder.encode(x)
    
    def forward(self, anchor, positive, negative):
        anchor_embed = self.forward_once(anchor)
        positive_embed = self.forward_once(positive)
        negative_embed = self.forward_once(negative)
        return anchor_embed, positive_embed, negative_embed

# Loss functions
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.mse_loss = nn.MSELoss()
    
    def forward(self, output, target):
        # For Siamese network - MSE between predicted and target similarity
        return self.mse_loss(output, target)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, 2)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, 2)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Training functions
def train_siamese(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        input1, input2, target = batch
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(input1, input2)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def train_triplet(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        anchor, positive, negative = batch
        
        optimizer.zero_grad()
        anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

# Evaluation functions
def evaluate(model, queries, documents, citation_graph, top_k=[1, 3, 5]):
    model.eval()
    all_scores = []
    all_labels = []
    
    # Encode all documents once
    doc_embeddings = model.encoder.encode(documents)
    
    # Calculate MRR and F1@k
    mrr = 0.0
    f1_scores = {k: 0.0 for k in top_k}
    
    for q_idx, query in enumerate(queries):
        # Encode query
        query_embedding = model.encoder.encode([query])[0]
        
        # Calculate similarities
        similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        
        # Get ground truth (documents cited by this query)
        try:
            true_positives = list(citation_graph.successors(q_idx))
        except:
            true_positives = []
        
        # Sort documents by similarity
        sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        
        # Calculate MRR
        for rank, doc_idx in enumerate(sorted_indices, 1):
            if doc_idx in true_positives:
                mrr += 1.0 / rank
                break
        
        # Calculate F1@k
        for k in top_k:
            predicted_positives = set(sorted_indices[:k])
            actual_positives = set(true_positives)
            
            tp = len(predicted_positives & actual_positives)
            fp = len(predicted_positives - actual_positives)
            fn = len(actual_positives - predicted_positives)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores[k] += f1
    
    # Normalize metrics
    num_queries = len(queries)
    mrr /= num_queries
    for k in top_k:
        f1_scores[k] /= num_queries
    
    return mrr, f1_scores

# Submodular optimization for reference recommendation
class SubmodularScorer:
    def __init__(self, model, documents, citation_graph, alpha=0.4):
        self.model = model
        self.documents = documents
        self.citation_graph = citation_graph
        self.alpha = alpha
        self.doc_embeddings = None
        self.clusters = self._cluster_documents()
    
    def _cluster_documents(self):
        # Simple clustering based on citation graph communities
        # In practice, you might use more sophisticated clustering
        return list(nx.algorithms.community.greedy_modularity_communities(self.citation_graph))
    
    def _compute_rewards(self, query_embedding, candidate_indices):
        rewards = []
        for doc_idx in candidate_indices:
            # Reward based on similarity to query
            doc_embedding = self.doc_embeddings[doc_idx]
            sim_score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            
            # Additional reward based on citation relationships
            try:
                path_length = nx.shortest_path_length(self.citation_graph, source=doc_idx, target=0)  # Assuming query is node 0
                citation_reward = self.alpha ** (path_length - 1)
            except:
                citation_reward = 0
            
            total_reward = sim_score + citation_reward
            rewards.append(total_reward)
        return rewards
    
    def score_function(self, R):
        """Submodular scoring function"""
        score = 0.0
        for cluster in self.clusters:
            cluster_docs = [doc_idx for doc_idx in R if doc_idx in cluster]
            if not cluster_docs:
                continue
            # Sum of rewards for documents in this cluster
            sum_rewards = sum(self.rewards[doc_idx] for doc_idx in cluster_docs)
            score += np.sqrt(sum_rewards)
        return score
    
    def recommend(self, query, M=5):
        """Greedy algorithm for submodular maximization"""
        # Encode query and documents if not already done
        if self.doc_embeddings is None:
            self.doc_embeddings = self.model.encoder.encode(self.documents)
        
        query_embedding = self.model.encoder.encode([query])[0]
        
        # Compute rewards for all documents
        candidate_indices = list(range(len(self.documents)))
        self.rewards = self._compute_rewards(query_embedding, candidate_indices)
        
        # Initialize
        R = set()
        remaining = set(candidate_indices)
        
        for _ in range(M):
            if not remaining:
                break
            
            best_doc = None
            best_gain = -float('inf')
            
            for doc_idx in remaining:
                current_R = R.copy()
                current_R.add(doc_idx)
                gain = self.score_function(current_R) - self.score_function(R)
                
                if gain > best_gain:
                    best_gain = gain
                    best_doc = doc_idx
            
            if best_doc is not None:
                R.add(best_doc)
                remaining.remove(best_doc)
        
        return list(R)

# Main training and evaluation pipeline
def main():
    config = Config()
    
    # Load and prepare data (placeholder - replace with actual data loading)
    # In practice, you would load your queries, documents, and citation graph
    #queries =  # List of query texts
    #documents =  # List of document texts
    citation_graph = nx.DiGraph()          # Citation graph (nodes are document indices)
    
    # Generate positive and negative samples based on citation graph
    positives = defaultdict(list)
    negatives = defaultdict(list)
    
    for q_idx in range(len(queries)):
        # Positive samples: documents cited by this query (distance < 5)
        try:
            positives[q_idx] = [n for n in citation_graph.successors(q_idx) 
                              if nx.shortest_path_length(citation_graph, q_idx, n) < 5]
        except:
            positives[q_idx] = []
        
        # Negative samples: random documents not cited by this query
        all_docs = set(range(len(documents)))
        cited_docs = set(positives[q_idx])
        non_cited = list(all_docs - cited_docs)
        negatives[q_idx] = random.sample(non_cited, min(len(positives[q_idx]), len(non_cited)))
    
    # Initialize models
    siamese_model = SiameseNetwork(config.embedding_dim).to(config.device)
    triplet_model = TripletNetwork(config.embedding_dim).to(config.device)
    
    # Initialize loss functions and optimizers
    siamese_criterion = ContrastiveLoss()
    triplet_criterion = TripletLoss(margin=config.margin)
    
    siamese_optimizer = optim.Adam(siamese_model.parameters(), lr=config.learning_rate)
    triplet_optimizer = optim.Adam(triplet_model.parameters(), lr=config.learning_rate)
    
    # Create datasets
    siamese_dataset = SiameseDataset(queries, positives, negatives, citation_graph, config.alpha)
    triplet_dataset = TripletDataset(queries, positives, negatives, citation_graph)
    
    siamese_loader = DataLoader(siamese_dataset, batch_size=config.batch_size, shuffle=True)
    triplet_loader = DataLoader(triplet_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop for Siamese network
    print("Training Siamese Network...")
    for distance, epochs in config.epochs_siamese.items():
        print(f"Training with distance={distance} for {epochs} epochs")
        for epoch in range(epochs):
            loss = train_siamese(siamese_model, siamese_loader, siamese_criterion, 
                                siamese_optimizer, config.device)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    # Training loop for Triplet network
    print("\nTraining Triplet Network...")
    for distance, epochs in config.epochs_triplet.items():
        print(f"Training with distance={distance} for {epochs} epochs")
        for epoch in range(epochs):
            loss = train_triplet(triplet_model, triplet_loader, triplet_criterion, 
                               triplet_optimizer, config.device)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    # Evaluation
    print("\nEvaluating Siamese Network...")
    siamese_mrr, siamese_f1 = evaluate(siamese_model, queries, documents, citation_graph, config.top_k)
    print(f"Siamese Network - MRR: {siamese_mrr:.3f}, F1@k: {siamese_f1}")
    
    print("\nEvaluating Triplet Network...")
    triplet_mrr, triplet_f1 = evaluate(triplet_model, queries, documents, citation_graph, config.top_k)
    print(f"Triplet Network - MRR: {triplet_mrr:.3f}, F1@k: {triplet_f1}")
    
    # Submodular recommendation
    print("\nTesting Submodular Recommendation...")
    scorer = SubmodularScorer(siamese_model, documents, citation_graph, config.alpha)
    
    # Test with a sample query
    sample_query = queries[0]
    recommendations = scorer.recommend(sample_query, M=5)
    print(f"Query: {sample_query}")
    print("Recommended references:")
    for i, doc_idx in enumerate(recommendations, 1):
        print(f"{i}. {documents[doc_idx][:100]}...")

if __name__ == "__main__":
    main()