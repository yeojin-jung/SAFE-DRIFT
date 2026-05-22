import numpy as np
import json
import hashlib
from itertools import islice
import re

try:
    from nltk.tokenize import WordPunctTokenizer
except ModuleNotFoundError:
    class WordPunctTokenizer:  # type: ignore[override]
        def tokenize(self, text):
            return re.findall(r"\w+|[^\w\s]", text)

wpt = WordPunctTokenizer()

def hash_buckets(string, num_buckets=10000):
    digest = hashlib.sha1(string.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_buckets

def get_ngram_counts(text, n=2, num_buckets=10000):
    """Generates hashed n-gram counts for a string."""
    words = wpt.tokenize(text.lower())
    unigrams = words
    bigrams = list(zip(words, islice(words, 1, None)))

    counts = np.zeros(num_buckets, dtype=int)
    for u in unigrams:
        counts[hash_buckets(u, num_buckets)] += 1
    for b in bigrams:
        # bigram is a tuple, convert to string for hashing
        counts[hash_buckets(f"{b[0]} {b[1]}", num_buckets)] += 1
    return counts

def select_dsir(pool_jsonl_path, target_jsonl_path, k, num_buckets=10000, q_sample_ratio=0.3, q_seed=124):
    """
    DSIR Implementation:
    1. Compute target n-gram distribution (p)
    2. Compute pool n-gram distribution (q) from a pool sample
    3. Resample pool samples based on Importance Weight w = p/q
    """
    # 1. Target Distribution (p)
    # TODO: We should use a different dataset for learning q and evaluating
    target_counts = np.zeros(num_buckets)
    with open(target_jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Combine instruction + output for full content profile
            text = f"{data['instruction']} {data['output']}"
            target_counts += get_ngram_counts(text, num_buckets=num_buckets)
    
    p = target_counts / (target_counts.sum() + 1e-10)
    
    # 2. Pool Distribution (q)
    pool_counts = np.zeros(num_buckets)
    if q_sample_ratio is None or q_sample_ratio >= 1.0:
        with open(pool_jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                text = f"{data['instruction']} {data['output']}"
                pool_counts += get_ngram_counts(text, num_buckets=num_buckets)
    else:
        if q_sample_ratio <= 0:
            raise ValueError("q_sample_ratio must be in (0, 1].")

        with open(pool_jsonl_path, "r") as f:
            n_pool = sum(1 for _ in f)
            
        q_sample_size = max(1, int(n_pool * q_sample_ratio))
        rng = np.random.default_rng(q_seed)
        reservoir = []
        with open(pool_jsonl_path, "r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                text = f"{data['instruction']} {data['output']}"
                if idx < q_sample_size:
                    reservoir.append(text)
                else:
                    j = rng.integers(0, idx + 1)
                    if j < q_sample_size:
                        reservoir[j] = text
        for text in reservoir:
            pool_counts += get_ngram_counts(text, num_buckets=num_buckets)
            
    q = pool_counts / (pool_counts.sum() + 1e-10)
    
    # 3. Compute Log-Ratios (Importance Weights)
    # log(p/q) = log(p) - log(q)
    log_diff = np.log(p + 1e-8) - np.log(q + 1e-8)
    
    # 4. Score each pool sample
    scores = []
    # Re-reading or using stored pool_texts to save memory
    with open(pool_jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            text = f"{data['instruction']} {data['output']}"
            curr_ngram = get_ngram_counts(text, num_buckets=num_buckets)
            # Log-importance weight for the sample
            score = np.dot(curr_ngram, log_diff)
            scores.append(score)
            
    scores = np.array(scores)
    
    # DSIR Resampling: Select top k based on importance scores
    # (Note: DSIR often uses Gumbel noise for sampling, but for 
    # fixed-budget k, Top-K is common for deterministic comparison)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > len(scores):
        raise ValueError(f"k={k} exceeds pool size {len(scores)}.")
    chosen_idxs = np.argpartition(-scores, k - 1)[:k]
    
    return chosen_idxs
