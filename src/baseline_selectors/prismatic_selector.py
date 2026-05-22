from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .prismatic_cluster import ClusterManager


def select_prismatic(
    candidate_gradients: torch.Tensor,
    subset_budget: int,
    cluster_ratio: float = 0.1,
    sparsity: float = 0.5,
    num_iters: int = 20,
    method: str = "hard",
    seed: int = 42,
) -> list[int]:
    if candidate_gradients.ndim != 2:
        raise ValueError("candidate_gradients must be a 2D tensor [num_examples, gradient_dim].")
    if subset_budget <= 0:
        raise ValueError("subset_budget must be positive.")
    if candidate_gradients.shape[0] < subset_budget:
        raise ValueError("subset_budget cannot exceed the number of candidate examples.")
    if not (0 < cluster_ratio <= 1):
        raise ValueError("cluster_ratio must be in (0, 1].")
    if not (0 <= sparsity <= 1):
        raise ValueError("sparsity must be in [0, 1].")
    if method not in {"hard", "soft"}:
        raise ValueError("method must be either 'hard' or 'soft'.")

    gradients = candidate_gradients.float()
    n_pool = gradients.shape[0]
    n_clusters = min(n_pool, max(1, int(n_pool * cluster_ratio)))

    cluster_labels, cluster_centroids = ClusterManager.cluster_kmeans(
        gradients,
        k=n_clusters,
        num_iter=num_iters,
        use_tqdm=False,
    )
    cluster_labels = torch.argmax(F.normalize(gradients, dim=1) @ cluster_centroids.T, dim=-1)

    if method == "hard":
        generator = torch.Generator(device=gradients.device)
        generator.manual_seed(int(seed))
        small_clusters = ClusterManager.smallest_clusters(cluster_labels, sparsity)
        small_clusters_tensor = torch.tensor(small_clusters, device=cluster_labels.device)
        sparse_ids = torch.where(torch.isin(cluster_labels, small_clusters_tensor))[0]

        if len(sparse_ids) >= subset_budget:
            perm = torch.randperm(len(sparse_ids), device=gradients.device, generator=generator)[:subset_budget]
            return sparse_ids[perm].cpu().tolist()

        remaining = torch.where(~torch.isin(cluster_labels, small_clusters_tensor))[0]
        fill_count = subset_budget - len(sparse_ids)
        fill_perm = torch.randperm(len(remaining), device=gradients.device, generator=generator)[:fill_count]
        return torch.cat([sparse_ids, remaining[fill_perm]]).cpu().tolist()

    labels_cpu = cluster_labels.cpu().numpy()
    counts = np.bincount(labels_cpu, minlength=n_clusters)
    inv_counts = 1.0 / np.maximum(counts, 1)
    probs = inv_counts[labels_cpu]
    probs = probs / probs.sum()
    indices = np.random.default_rng(int(seed)).choice(n_pool, subset_budget, replace=False, p=probs)
    return [int(index) for index in indices.tolist()]
