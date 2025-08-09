"""
reference_recommendation.py

"Reference Recommendation for LLM-Generated Text Using Deep Textual Representations"

"""

from typing import List, Dict, Tuple, Iterable
import os
import json
import random
import math
import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Optional Faiss for efficient nearest neighbors (if installed)
try:
    import faiss
    _HAS_FAISS = True
except:
    _HAS_FAISS = False

# ---------------------------
# 
# ---------------------------
CONFIG = {
    "sentbert_model": "all-MiniLM-L6-v2",  # can use "sentence-transformers/..." or a distilBERT-based SBERT
    "learning_rate": 2e-5,
    "batch_size": 16,
    "siamese_alpha": 0.4,   # paper tuned alpha to 0.4
    "triplet_margin": 1.0,
    # epochs for distances (d=1..4)
    "siamese_epochs_per_distance": {1:30, 2:20, 3:10, 4:5},
    "triplet_epochs_per_distance": {1:25, 2:15, 3:7, 4:4},
    "max_distance_positive": 4,  # select positives with path distance <= 4
    "embedding_cache_file": "embeddings.npy",  # optional caching
    "use_faiss": _HAS_FAISS,
    "subref_clusters": 20,  # K for clustering when author clusters not available
}

# ---------------------------
# Data format expectations
# ---------------------------
# Corpus: list of dicts, each dict at least:
# {
#   'doc_id': str or int,
#   'title': str,
#   'text': str,         # full text used for embedding or a summary/abstract
#   'authors': [ ... ],  # optional list of author names (for author-based clusters)
#   'citations': [doc_id,...]  # list of doc_ids that this doc cites (directed edges)
# }
#
# Query set: list of dicts:
# {
#   'query_id': id,
#   'text': "LLM generated answer or conversation snippet",
#   'relevant_doc_ids': [doc_id,...]  # ground-truth references (from citation list if evaluating)
# }

# ---------------------------
# Utilities
# ---------------------------

def build_citation_graph(corpus: List[Dict]) -> nx.DiGraph:
    """Build directed citation graph from corpus metadata 'citations'."""
    G = nx.DiGraph()
    for doc in corpus:
        G.add_node(doc['doc_id'])
    for doc in corpus:
        from_id = doc['doc_id']
        for cited in doc.get('citations', []):
            if not G.has_node(cited):
                # optionally add cited node (could be outside collection)
                G.add_node(cited)
            G.add_edge(from_id, cited)
    return G

def compute_shortest_distances(G: nx.DiGraph, max_dist: int = 4) -> Dict[Tuple, int]:
    """
    Compute shortest path lengths between all nodes up to max_dist.
    Returns dict keyed by (u,v) -> distance (if <= max_dist), else omitted.
    """
    distances = {}
    # For each node, run BFS / shortest_path_length upto cutoff
    for node in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=max_dist)
        for v, d in lengths.items():
            if node != v and d <= max_dist:
                distances[(node, v)] = d
    return distances

# ---------------------------
# Sample generation
# ---------------------------

def get_embedding_index(embeddings: np.ndarray):
    """Return an index structure for nearest neighbor search. Uses faiss if available."""
    if CONFIG['use_faiss']:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product (for cosine we normalize)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return ("faiss", index)
    else:
        nbrs = NearestNeighbors(n_neighbors=50, metric='cosine').fit(embeddings)
        return ("sklearn", nbrs)

def knn_candidates(embedding: np.ndarray, embeddings: np.ndarray, index, top_k=50):
    """Return indices of nearest neighbors (excluding self)."""
    if index[0] == "faiss":
        _, I = index[1].search(embedding.reshape(1,-1).astype(np.float32), top_k+1)
        idxs = I[0].tolist()
        # faiss normalized inner-product returns cos sim ordering; remove self if present
        return idxs[1:] if idxs[0] == 0 else idxs[:top_k]
    else:
        distances, idxs = index[1].kneighbors(embedding.reshape(1,-1), n_neighbors=top_k+1)
        idxs = idxs[0].tolist()
        return idxs[1:] if idxs[0] == 0 else idxs[:top_k]

def create_training_samples(
    corpus: List[Dict],
    corpus_embeddings: np.ndarray,
    distances: Dict[Tuple, int],
    docid_to_index: Dict,
    negative_strategy: str = "most_dissimilar",  # one of 'random','closest','most_dissimilar','mix'
    seed: int = 42
) -> Dict[int, Dict]:
    """
    Create training examples for each query document:
      - positives: docs with distance <= CONFIG['max_distance_positive']
      - negatives: depends on strategy
    Returns mapping query_index -> {'positives': [idx,...], 'negatives': [idx,...]}
    """
    random.seed(seed)
    np.random.seed(seed)
    n = len(corpus)
    results = {}
    # Precompute nearest/farthest neighbor ordering for each doc using embeddings
    # We'll compute cosine similarities
    emb_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-12)
    sim_matrix = emb_norm @ emb_norm.T  # may be large; for large corpora use approximate NN
    for i, doc in enumerate(corpus):
        qid = doc['doc_id']
        # gather positives by distances dict (q -> others)
        positives = []
        for j in range(n):
            pair = (qid, corpus[j]['doc_id'])
            if pair in distances and distances[pair] <= CONFIG['max_distance_positive']:
                positives.append(j)
        # sample negatives
        if negative_strategy == 'random':
            negatives = [j for j in range(n) if j not in positives and j != i]
            # sample same number as positives
            negatives = random.sample(negatives, min(len(positives), len(negatives))) if positives else []
        else:
            sims = sim_matrix[i]
            # get sorted indices by similarity descending
            sorted_idx_desc = np.argsort(-sims)
            # exclude self and positives
            filtered = [idx for idx in sorted_idx_desc if idx != i and idx not in positives]
            if negative_strategy == 'closest':
                negatives = filtered[:len(positives)]
            elif negative_strategy == 'most_dissimilar':
                negatives = filtered[-len(positives):] if len(positives) > 0 else []
            elif negative_strategy == 'mix':
                npos = len(positives)
                half = npos // 2
                neg_closest = filtered[:half]
                neg_farthest = filtered[-(npos-half):] if npos-half>0 else []
                negatives = neg_closest + neg_farthest
            else:
                raise ValueError("unknown negative strategy")
        results[i] = {'positives': positives, 'negatives': negatives, 'query_index': i}
    return results

# ---------------------------
# Datasets for PyTorch
# ---------------------------

class SiamesePairsDataset(Dataset):
    def __init__(self, corpus_texts: List[str], samples: Dict[int, Dict], docid_to_index: Dict, alpha=0.4):
        """
        For Siamese: generate pairs (q, d) and a target similarity value.
        target similarity for positives per paper: sim(di,dj) = alpha * path^{-1}  (we use path from distances)
        Here we assume samples dict includes distance info if needed; for simplicity set positives target=alpha*(1)
        """
        self.corpus_texts = corpus_texts
        self.pairs = []  # list of (q_text, cand_text, target_sim)
        # For simplicity we use target 1 for positive scaled by alpha, 0 for negative
        for qidx, info in samples.items():
            qtext = corpus_texts[qidx]
            for p in info['positives']:
                t = alpha  # the paper uses alpha*(path^-1); for simplicity we use alpha (tunable)
                self.pairs.append((qtext, corpus_texts[p], float(t)))
            for n in info['negatives']:
                self.pairs.append((qtext, corpus_texts[n], float(0.0)))
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

class TripletDataset(Dataset):
    def __init__(self, corpus_texts: List[str], samples: Dict[int, Dict]):
        """Triplets of (query_text, positive_text, negative_text)"""
        self.triplets = []
        for qidx, info in samples.items():
            qtext = corpus_texts[qidx]
            positives = info['positives']
            negatives = info['negatives']
            # pair positives and negatives 1-to-1
            for p, n in zip(positives, negatives):
                self.triplets.append((qtext, corpus_texts[p], corpus_texts[n]))
    def __len__(self):
        return len(self.triplets)
    def __getitem__(self, idx):
        return self.triplets[idx]

# ---------------------------
# Model wrappers (SentenceTransformer fine-tuning)
# ---------------------------

class SiameseFineTuner:
    """
    Fine-tune a SentenceTransformer model in a siamese manner using MSE on cosine similarity.
    We'll use SentenceTransformer encode and compute cosine similarity of embeddings;
    then minimize MSE between predicted sim and target.
    """
    def __init__(self, model_name=CONFIG['sentbert_model'], device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)

    def train(self, dataset: SiamesePairsDataset, epochs=1, lr=2e-5, batch_size=16, save_path=None):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        mse = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"Siamese epoch {epoch+1}/{epochs}"):
                # batch is list of tuples -> collate manually
                q_texts, c_texts, targets = zip(*batch)
                q_emb = torch.tensor(self.model.encode(list(q_texts), convert_to_numpy=True)).to(self.device)
                c_emb = torch.tensor(self.model.encode(list(c_texts), convert_to_numpy=True)).to(self.device)
                # normalize for cosine
                q_norm = torch.nn.functional.normalize(q_emb, p=2, dim=1)
                c_norm = torch.nn.functional.normalize(c_emb, p=2, dim=1)
                preds = torch.sum(q_norm * c_norm, dim=1)  # cosine similarity in [-1,1]
                # targets are in [0,1] (alpha or 0). Ensure same range; if negative similarity, clamp to [-1,1]
                t = torch.tensor(targets, dtype=preds.dtype, device=self.device)
                loss = mse(preds, t)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item() * len(targets)
            print(f"Epoch {epoch+1} loss: {total_loss/len(dataset):.6f}")
        if save_path:
            self.model.save(save_path)
        return self.model

class TripletFineTuner:
    """
    Fine-tune using triplet loss (hinge) on SentenceTransformer embeddings.
    """
    def __init__(self, model_name=CONFIG['sentbert_model'], device=None, margin=1.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginWithDistance(margin=self.margin, distance_function=lambda x,y: 1 - torch.nn.functional.cosine_similarity(x,y), reduction='mean')

    def train(self, dataset: TripletDataset, epochs=1, lr=2e-5, batch_size=16, save_path=None):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"Triplet epoch {epoch+1}/{epochs}"):
                q_texts, p_texts, n_texts = zip(*batch)
                q_emb = torch.tensor(self.model.encode(list(q_texts), convert_to_numpy=True)).to(self.device)
                p_emb = torch.tensor(self.model.encode(list(p_texts), convert_to_numpy=True)).to(self.device)
                n_emb = torch.tensor(self.model.encode(list(n_texts), convert_to_numpy=True)).to(self.device)
                loss = self.triplet_loss(q_emb, p_emb, n_emb)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item() * len(q_texts)
            print(f"Epoch {epoch+1} loss: {total_loss/len(dataset):.6f}")
        if save_path:
            self.model.save(save_path)
        return self.model

# ---------------------------
# Retrieval & Submodular selection
# ---------------------------

def embed_corpus(model: SentenceTransformer, corpus_texts: List[str], batch_size=64, cache_path=None):
    """Compute embeddings for all documents and optionally cache to disk."""
    if cache_path and os.path.exists(cache_path):
        print("Loading embeddings from cache", cache_path)
        return np.load(cache_path)
    emb = model.encode(corpus_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    if cache_path:
        np.save(cache_path, emb)
    return emb

def retrieve_by_sentbert(
    model: SentenceTransformer,
    query_text: str,
    corpus_embeddings: np.ndarray,
    corpus_texts: List[str],
    top_k: int = 10,
    use_faiss: bool = CONFIG['use_faiss'],
):
    """Retrieve top_k docs for single query using SentenceTransformer cosine similarity."""
    q_emb = model.encode(query_text, convert_to_numpy=True)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    emb_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-12)
    sims = (emb_norm @ q_norm).ravel()
    idxs = np.argsort(-sims)[:top_k]
    return idxs, sims[idxs]

def subref_greedy_select(
    candidate_indices: List[int],
    query_sim_scores: Dict[int, float],
    corpus_clusters: Dict[int, int],
    K: int = 5
) -> List[int]:
    """
    Greedy submodular selection following Eq (4):
    f(A) = sum_k sqrt(sum_{l in A∩Ck} R_{k,l})
    We'll implement R_{k,l} as the similarity score between doc l and the query.
    candidate_indices: list of doc indices to consider (e.g., top-N by similarity)
    query_sim_scores: dict idx -> sim score
    corpus_clusters: dict idx -> cluster_id (0..K-1)
    """
    selected = []
    # Precompute cluster membership
    # cluster_rewards[k] = list of (idx, reward)
    Kclusters = max(corpus_clusters.values())+1
    cluster_selected_reward = {k: 0.0 for k in range(Kclusters)}
    remaining = set(candidate_indices)
    while len(selected) < K and remaining:
        best_gain = -1e9
        best_idx = None
        for idx in list(remaining):
            k = corpus_clusters[idx]
            # current cluster sum = cluster_selected_reward[k]
            new_sum = cluster_selected_reward[k] + query_sim_scores.get(idx, 0.0)
            # marginal gain for cluster k is sqrt(new_sum)-sqrt(cluster_selected_reward[k])
            gain = math.sqrt(new_sum) - math.sqrt(cluster_selected_reward[k])
            # total gain across clusters equals this gain (only affects one cluster)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None:
            break
        # select it
        selected.append(best_idx)
        k = corpus_clusters[best_idx]
        cluster_selected_reward[k] += query_sim_scores.get(best_idx, 0.0)
        remaining.remove(best_idx)
    return selected

def build_clusters_for_subref(corpus: List[Dict], corpus_embeddings: np.ndarray, K=20):
    """
    Return mapping doc_index -> cluster_id
    Prefer using author-based clustering if author metadata exists (combine authors into keys).
    Otherwise apply KMeans over embeddings (paper suggests using author clusters).
    """
    # check author metadata
    author_map = {}
    has_authors = all('authors' in d and d['authors'] for d in corpus)
    if has_authors:
        # make cluster per author (or cluster authors) - to keep it simple, map unique author name -> id
        cluster_id = {}
        idx_to_cluster = {}
        next_c = 0
        for i, d in enumerate(corpus):
            # combine authors into a key (first author or join)
            first_author = d['authors'][0] if d['authors'] else "unknown"
            if first_author not in cluster_id:
                cluster_id[first_author] = next_c
                next_c += 1
            idx_to_cluster[i] = cluster_id[first_author]
        return idx_to_cluster
    else:
        # KMeans clustering of embeddings
        k = min(K, len(corpus))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(corpus_embeddings)
        labels = kmeans.labels_
        return {i: int(labels[i]) for i in range(len(corpus))}

# ---------------------------
# Evaluation metrics
# ---------------------------

def f1_at_k(preds: List[int], gold: List[int], k: int) -> float:
    topk = preds[:k]
    tp = len(set(topk) & set(gold))
    precision = tp / k
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def mrr_single(preds: List[int], gold: List[int]) -> float:
    for i, p in enumerate(preds, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0

# ---------------------------
# Example end-to-end usage
# ---------------------------

def example_pipeline(corpus: List[Dict], queries: List[Dict], train=True):
    """
    Example pipeline showing:
     - build graph, distances
     - compute embeddings
     - generate training samples
     - fine-tune siamese or triplet
     - perform retrieval and subref selection
     - evaluate F1@1/3/5 and MRR
    """
    # 1) Build map docid <-> index
    docid_to_index = {d['doc_id']: i for i, d in enumerate(corpus)}
    index_to_docid = {i: d['doc_id'] for i, d in enumerate(corpus)}
    corpus_texts = [d['text'] for d in corpus]

    # 2) Build citation graph and compute distances
    G = build_citation_graph(corpus)
    print("Citation graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    distances = compute_shortest_distances(G, max_dist=CONFIG['max_distance_positive'])
    print("Computed pair distances up to", CONFIG['max_distance_positive'])

    # 3) Initialize SBERT and compute embeddings (or load cache)
    bert = SentenceTransformer(CONFIG['sentbert_model'])
    corpus_embeddings = embed_corpus(bert, corpus_texts, cache_path=None)

    # 4) Create training samples (choose negative strategy, e.g., 'most_dissimilar')
    samples = create_training_samples(corpus, corpus_embeddings, distances, docid_to_index, negative_strategy='most_dissimilar')

    # 5) Optionally fine-tune:
    if train:
        # Siamese fine-tuning as in paper with epoch schedule per distance:
        siamese = SiameseFineTuner(model_name=CONFIG['sentbert_model'])
        # For simplicity, aggregate all distances; in the paper they loop per distance with epoch schedules.
        siamese_dataset = SiamesePairsDataset(corpus_texts, samples, docid_to_index, alpha=CONFIG['siamese_alpha'])
        print("Starting Siamese fine-tuning (this can be slow).")
        siamese_model = siamese.train(siamese_dataset, epochs=3, lr=CONFIG['learning_rate'], batch_size=CONFIG['batch_size'])  # reduced epochs for example
        # Or Triplet fine-tuning:
        triplet = TripletFineTuner(model_name=CONFIG['sentbert_model'], margin=CONFIG['triplet_margin'])
        triplet_dataset = TripletDataset(corpus_texts, samples)
        print("Starting Triplet fine-tuning (this can be slow).")
        triplet_model = triplet.train(triplet_dataset, epochs=3, lr=CONFIG['learning_rate'], batch_size=CONFIG['batch_size'])

        # choose which to use; for now use siamese_model
        model_to_use = siamese_model
    else:
        model_to_use = bert

    # 6) Build clusters required for SubRef (author-based preferred)
    clusters = build_clusters_for_subref(corpus, corpus_embeddings, K=CONFIG['subref_clusters'])

    # 7) Retrieval + SubRef for each query and evaluate
    all_f1_1, all_f1_3, all_f1_5, all_mrr = [], [], [], []
    for q in tqdm(queries, desc="Evaluate queries"):
        qtext = q['text']
        # get top 200 candidate by SentBERT
        idxs, sims = retrieve_by_sentbert(model_to_use, qtext, corpus_embeddings, corpus_texts, top_k=200)
        # prepare scores and cluster mapping
        candidate_indices = idxs.tolist()
        sim_dict = {int(idx): float(sim) for idx, sim in zip(idxs, sims)}
        # run subref greedy to pick top 5
        selected = subref_greedy_select(candidate_indices, sim_dict, clusters, K=5)
        # we also consider plain SentBERT ranking top-k
        plain_preds = candidate_indices
        # Evaluate: gold doc indices (map doc_ids)
        gold = [docid_to_index[g] for g in q.get('relevant_doc_ids', []) if g in docid_to_index]
        # compute metrics for SentBERT plain
        all_f1_1.append(f1_at_k(plain_preds, gold, 1))
        all_f1_3.append(f1_at_k(plain_preds, gold, 3))
        all_f1_5.append(f1_at_k(plain_preds, gold, 5))
        all_mrr.append(mrr_single(plain_preds, gold))
        # You can compute the same for subref-selected as well if desired
    # aggregate
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    print("SentBERT results (mean): F1@1:", mean(all_f1_1), "F1@3:", mean(all_f1_3), "F1@5:", mean(all_f1_5), "MRR:", mean(all_mrr))


if __name__ == "__main__":
    # toy data example (very small)
    corpus = [
        {'doc_id': 'A', 'text': "Deep learning for document classification", 'authors': ['Author1'], 'citations': ['B']},
        {'doc_id': 'B', 'text': "Convolutional neural networks for video clustering", 'authors': ['Author2'], 'citations': ['C']},
        {'doc_id': 'C', 'text': "Foundation of Data Science", 'authors': ['Author3'], 'citations': []},
        {'doc_id': 'D', 'text': "Cyberscurity foundation and trends", 'authors': ['Author4'], 'citations': []},
    ]
    # use citation edges so distances: A->B (1), A->C (2) etc.
    queries = [
        {'query_id': 'q1', 'text': "...", 'relevant_doc_ids': ['Doc1']},
        {'query_id': 'q2', 'text': "...", 'relevant_doc_ids': ['Doc2']},
    ]
    example_pipeline(corpus, queries, train=False)
