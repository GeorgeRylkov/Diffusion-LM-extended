"""
Test rounding accuracy of GPT-2 token embeddings after PCA reduction.
Compares raw PCA vs z-scored PCA at 128d/256d, plus a raw 768d baseline.
"""
import numpy as np
import torch
from transformers import GPT2Model
from sklearn.decomposition import PCA


def rounding_accuracy_batched(W_reduced, sigma, n_test=5000, batch_size=500):
    """
    Test rounding accuracy: add noise to a sample of embeddings,
    find nearest neighbor, check if it's the original token.
    """
    n_tokens = W_reduced.shape[0]
    test_indices = np.random.choice(n_tokens, size=min(n_test, n_tokens), replace=False)
    W_torch = torch.tensor(W_reduced, dtype=torch.float32)
    correct = 0
    total = 0
    for start in range(0, len(test_indices), batch_size):
        batch_idx = test_indices[start:start + batch_size]
        originals = W_torch[batch_idx]
        noise = torch.randn_like(originals) * sigma
        noisy = originals + noise
        dists = torch.cdist(noisy, W_torch, p=2)
        predicted = dists.argmin(dim=1).numpy()
        correct += (predicted == batch_idx).sum()
        total += len(batch_idx)
    return correct / total


def zscore(W):
    """Per-dimension zero-mean unit-variance normalization."""
    mean = W.mean(axis=0)
    std = np.clip(W.std(axis=0), 1e-6, None)
    return (W - mean) / std


def run_suite(label, W, noise_sigmas, n_test):
    print(f"\n{'='*70}")
    print(label)
    print(f"  shape: {W.shape}, mean norm: {np.linalg.norm(W, axis=-1).mean():.4f}")
    print(f"{'='*70}")
    for sigma in noise_sigmas:
        acc = rounding_accuracy_batched(W, sigma, n_test=n_test)
        print(f"  sigma={sigma:.1f}: accuracy = {acc:.4f}")


def main():
    print("Loading GPT-2...")
    model = GPT2Model.from_pretrained("gpt2")
    W = model.wte.weight.detach().numpy()  # [50257, 768]
    del model
    print(f"Token embedding matrix: {W.shape}")

    noise_sigmas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    n_test = 10000
    np.random.seed(42)
    torch.manual_seed(42)

    # ---- Baselines: full 768d ----
    run_suite("BASELINE: Raw GPT-2 768d (no processing)", W, noise_sigmas, n_test)
    run_suite("BASELINE: Z-scored GPT-2 768d", zscore(W), noise_sigmas, n_test)

    # ---- PCA reductions ----
    for dim in [128, 256]:
        pca = PCA(n_components=dim)
        W_pca = pca.fit_transform(W)  # PCA centers internally
        var = pca.explained_variance_ratio_.sum()

        run_suite(
            f"PCA to {dim}d (raw, {var*100:.1f}% variance)",
            W_pca, noise_sigmas, n_test)

        run_suite(
            f"PCA to {dim}d + z-score ({var*100:.1f}% variance)",
            zscore(W_pca), noise_sigmas, n_test)

    # ---- BERT-tiny 128d reference ----
    from transformers import BertModel
    print("\nLoading BERT-tiny...")
    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    W_bert = bert.embeddings.word_embeddings.weight.detach().numpy()
    del bert
    print(f"BERT-tiny embedding matrix: {W_bert.shape}")

    run_suite("REFERENCE: BERT-tiny 128d (raw)", W_bert, noise_sigmas, n_test)
    run_suite("REFERENCE: BERT-tiny 128d (z-scored)", zscore(W_bert), noise_sigmas, n_test)

    print("\nDone.")


if __name__ == "__main__":
    main()
