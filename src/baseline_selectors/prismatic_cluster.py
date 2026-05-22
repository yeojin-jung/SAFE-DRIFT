import random
from typing import Tuple, List

import torch
import torch.nn.functional as F
from tqdm import tqdm


class ClusterManager:
    @staticmethod
    def cluster_kmeans(data: torch.Tensor, k: int, num_iter: int, use_tqdm: bool = False) -> Tuple:
        """
        K-Means algorithm optimized for VRAM usage.
        Input:
            data: torch.Tensor, sized (n, grad_dim). data should be normalized.
            k: number of clusters
            num_iter: number of iterations for Lloyd's algorithm
        Output:
            labels: torch.Tensor, sized (n,). Cluster assignment for each point in `data`.
            centroids: torch.Tensor, sized (k, grad_dim). Centroids for the k clusters.
        """
        data = F.normalize(data, dim=1)

        # -- initialize centroids and centroid labels -- #
        centroids = data[:k, :].clone()  # (k, grad_dim)
        labels = None  # (n,)

        # -- loop in Lloyd's algorithm -- #
        for _ in tqdm(range(num_iter), disable=not use_tqdm, desc="Cluster-Kmeans"):
            # -- E step: assign points to the closest cluster according to cosine similarity -- #
            labels = ClusterManager._calculate_sim_matrix_and_label(data, centroids)

            # -- M step: update the centroids to the normalized cluster average -- #
            # compute the sum of samples per cluster - we don't use scatter_add_ to save memory :)
            centroids.zero_()
            for cluster_idx in range(k):
                centroids[cluster_idx] = torch.sum(data[labels == cluster_idx], dim=0)

            # normalize centroids
            centroids = F.normalize(centroids, dim=1)

        return labels, centroids

    @staticmethod
    def _calculate_sim_matrix_and_label(data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient subroutine for cluster_kmeans.
        Given data of size (n, embed_dim) and centroids (k, embed_size),
        (1) compute the similarity matrix sized (n, embed_dim),
        (2) compute the labels for each sample in data sized (n,)
        and return the labels (n,)
        """
        max_batch_num_centroids = 90

        max_values_list, max_indices_list = [], []
        for batch_start_idx in range(0, centroids.size(0), max_batch_num_centroids):
            batch_similarity_matrix = data @ centroids[
                                             batch_start_idx:batch_start_idx + max_batch_num_centroids].T  # (n, batch_num_centroids)
            max_values, max_indices = batch_similarity_matrix.max(dim=1)
            max_indices += batch_start_idx  # index should represent actual centroid index across batches
            max_values_list.append(max_values)
            max_indices_list.append(max_indices)

            del batch_similarity_matrix

        max_values_in_each_batch = torch.stack(max_values_list, dim=1)  # (n, num_batches)
        max_indices_in_each_batch = torch.stack(max_indices_list, dim=1)  # (n, num_batches)
        indices_of_max_values_across_batches = max_values_in_each_batch.argmax(
            dim=-1)  # (n,), where i-th element is the index of the batch whose max value was the largest across all batches
        labels = max_indices_in_each_batch[
            torch.arange(max_indices_in_each_batch.size(0), device=data.device),
            indices_of_max_values_across_batches,
        ]

        return labels

    @staticmethod
    def select_sampling_cluster(labels: torch.Tensor) -> int:
        """
        Pick 1 cluster to sample few-shot examples from.
        Random-sample 1 cluster among the smallest 10% clusters.
        Input:
            labels: torch.Tensor, sized (n,). Cluster assignment for each point in `data`.
            k: int, total number of clusters
        Output:
            selected_cluster: int, index of the selected cluster
        """
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)

        # compute size of each cluster
        cluster_sizes = torch.bincount(labels)

        # pick the smallest 10% clusters
        candidate_clusters = []  # list of candidate cluster indices
        threshold = torch.sort(cluster_sizes)[0][int(cluster_sizes.size(-1) * 0.1)]
        for cluster_idx, cluster_size in enumerate(cluster_sizes.tolist()):
            if cluster_size < threshold:
                candidate_clusters.append(cluster_idx)

        if not candidate_clusters:
            candidate_clusters = list(range(cluster_sizes.numel()))

        # sample 1 cluster from `candidate_clusters`
        selected_cluster = random.sample(candidate_clusters, 1)[0]

        return selected_cluster

    @staticmethod
    def smallest_clusters(labels: torch.Tensor, ratio: float) -> List[int]:
        """
        Return list of indices of clusters with top `ratio` small sizes.
        """
        assert 0 <= ratio <= 1, "`ratio` must be in 0 ~ 1."

        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)

        # compute size of each cluster
        cluster_sizes = torch.bincount(labels)

        # pick the smallest `ratio` clusters
        candidate_clusters = []  # list of candidate cluster indices
        threshold_index = min(cluster_sizes.size(-1) - 1, int(cluster_sizes.size(-1) * ratio))
        threshold = torch.sort(cluster_sizes)[0][threshold_index]
        for cluster_idx, cluster_size in enumerate(cluster_sizes.tolist()):
            if cluster_size < threshold:
                candidate_clusters.append(cluster_idx)

        if not candidate_clusters:
            candidate_clusters = list(range(cluster_sizes.numel()))

        return candidate_clusters

    @staticmethod
    def select_fewshot_examples(labels: torch.Tensor, max_num_fewshot_examples: int) -> List[int]:
        """
        Given cluster labels for existing samples, sample `num_few_shot_examples` examples.
        Input:
            labels: torch.Tensor, sized (n,). Cluster assignment for each point in `data`.
            num_fewshot_examples: int
        Output:
            selected_examples: List[int], list of selected sample indices in `labels`.
                               The number of fewshot examples may be smaller if the cluster is too small.
        """
        # select which cluster to sample examples from
        sampling_cluster = ClusterManager.select_sampling_cluster(labels)

        # random-sample fewshot examples from `sampling_cluster`
        members_in_sampling_cluster = (labels == sampling_cluster).nonzero().view(-1).tolist()

        num_examples_to_sample = min(len(members_in_sampling_cluster), max_num_fewshot_examples)
        selected_examples = random.sample(members_in_sampling_cluster, num_examples_to_sample)

        return selected_examples
