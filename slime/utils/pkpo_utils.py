"""
Pass-at-k Policy Optimization (PKPO) utilities.

Implements the reward batch transformations from:
  "Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems"
  (Walder & Karkhanis, NeurIPS 2025)

Given n i.i.d. samples per prompt with rewards g_1, ..., g_n, these functions
transform the reward vector R^n -> R^n so that the resulting policy gradient
is an unbiased estimator of the gradient of maxg@k (the expected maximum
reward over k independent samples).

The key functions are:
  - sloo: Leave-one-out baselined estimator (Equation 29)
  - sloo_minus_one: k-1 leave-one-out baselined estimator (Equation 33)
  - rho: Unbiased maxg@k estimator (Equation 12)
"""

from typing import List, Tuple

import torch


def _m_normed(N: int, K: int, i: int, j: int) -> float:
    """Compute normalized matrix element m_{ij} / C(N,K).

    Uses the stable product formulation (Equation 14) to avoid
    large binomial coefficients.
    """
    if i == j and i >= K - 1:
        # Diagonal: C(i-1, K-1) / C(N, K) via telescoping products
        num = torch.arange(i - K + 2, i + 1, dtype=torch.float64)
        den = torch.arange(N - K + 2, N + 1, dtype=torch.float64)
        return float(K / (N - K + 1) * (num / den).prod())
    elif j > i and j >= K - 1 and K >= 2:
        # Off-diagonal: C(j-2, K-2) / C(N, K) via telescoping products
        num = torch.arange(j - K + 2, j, dtype=torch.float64)
        den = torch.arange(N - K + 2, N, dtype=torch.float64)
        return float(K / (N - K + 1) * (K - 1) / N * (num / den).prod())
    return 0.0


def _m_diagonal(N: int, K: int) -> torch.Tensor:
    """Compute the diagonal of the normalized M matrix."""
    return torch.tensor([_m_normed(N, K, i, i) for i in range(N)], dtype=torch.float64)


def _delta(N: int, K: int, i: int) -> float:
    return _m_normed(N, K, i, i + 1) - _m_normed(N, K, i + 1, i + 1)


def _deltas(N: int, K: int) -> torch.Tensor:
    return torch.tensor([_delta(N - 1, K, i) for i in range(N - 2)], dtype=torch.float64)


def rho(g: torch.Tensor, K: int) -> float:
    """Unbiased estimator of maxg@k (Equation 12).

    Args:
        g: Reward vector of shape (N,).
        K: The k in pass@k.

    Returns:
        Scalar estimate of maxg@k.
    """
    g_sorted, _ = g.to(torch.float64).sort()
    diag = _m_diagonal(len(g), K).to(g.device)
    return float((g_sorted * diag).sum())


def _s_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    """Compute s_i (Equation 19) on already-sorted rewards.

    Returns tensor in sorted order.
    """
    N = len(g_sorted)
    g_sorted = g_sorted.to(torch.float64)
    diag = _m_diagonal(N, K).to(g_sorted.device)

    c = g_sorted * diag
    if N > 1:
        deltas = _deltas(N + 1, K).to(g_sorted.device)
        c[: N - 1] = c[: N - 1] + g_sorted[1:] * deltas
    return torch.cumsum(c.flip(0), dim=0).flip(0)


def s(g: torch.Tensor, K: int) -> torch.Tensor:
    """Unbiased gradient weights s_i (Equation 19).

    Args:
        g: Reward vector of shape (N,).
        K: The k in pass@k.

    Returns:
        Transformed reward vector of shape (N,).
    """
    i_sort = g.argsort()
    g_sorted = g[i_sort].to(torch.float64)
    result = torch.zeros_like(g, dtype=torch.float64)
    result[i_sort] = _s_sorted(g_sorted, K)
    return result.to(g.dtype)


def _b_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    """Compute LOO baseline b_i^{(K)} on sorted rewards (Equation 30-32)."""
    N = len(g_sorted)
    g_sorted = g_sorted.to(torch.float64)
    diag = _m_diagonal(N - 1, K).to(g_sorted.device)

    w = (diag * torch.arange(1, N, dtype=torch.float64, device=g_sorted.device))
    if N > 2:
        deltas = _deltas(N, K).to(g_sorted.device)
        w[1:] = w[1:] + deltas * torch.arange(1, N - 1, dtype=torch.float64, device=g_sorted.device)

    c1 = (w * g_sorted[1:]).sum().unsqueeze(0)
    c2 = (g_sorted[:-1] - g_sorted[1:]) * w
    return torch.cumsum(torch.cat([c1, c2]), dim=0)


def sloo(g: torch.Tensor, K: int) -> torch.Tensor:
    """LOO-baselined gradient weights s_i^{(loo)} (Equation 29).

    Args:
        g: Reward vector of shape (N,).
        K: The k in pass@k.

    Returns:
        Transformed reward vector of shape (N,).
    """
    N = len(g)
    i_sort = g.argsort()
    g_sorted = g[i_sort].to(torch.float64)

    s_vals = _s_sorted(g_sorted, K)
    b_vals = _b_sorted(g_sorted, K)
    result_sorted = s_vals - b_vals / (N - 1)

    result = torch.zeros_like(g, dtype=torch.float64)
    result[i_sort] = result_sorted
    return result.to(g.dtype)


def sloo_minus_one(g: torch.Tensor, K: int) -> torch.Tensor:
    """s_i^{(loo-1)}: k-1 LOO-baselined gradient weights (Equation 33).

    Uses subsets of size k-1 for the baseline, which is the recommended
    estimator from the paper due to lower variance.

    Args:
        g: Reward vector of shape (N,).
        K: The k in pass@k.

    Returns:
        Transformed reward vector of shape (N,).
    """
    N = len(g)
    i_sort = g.argsort()
    g_sorted = g[i_sort].to(torch.float64)

    s_vals = _s_sorted(g_sorted, K)
    b_vals = _b_sorted(g_sorted, K - 1)
    result_sorted = s_vals - b_vals * K / ((K - 1) * N)

    result = torch.zeros_like(g, dtype=torch.float64)
    result[i_sort] = result_sorted
    return result.to(g.dtype)


def pkpo_transform_rewards(
    rewards: torch.Tensor,
    n_samples_per_prompt: int,
    k: int,
    estimator_type: str = "sloo_minus_one",
) -> torch.Tensor:
    """Apply PKPO reward transformation to a batch of rewards.

    Groups rewards by prompt (n_samples_per_prompt samples each),
    applies the chosen PKPO estimator to each group, and returns
    the transformed rewards.

    Args:
        rewards: Flat tensor of all rewards in the batch [B].
        n_samples_per_prompt: Number of samples per prompt (n).
        k: The k in pass@k to optimize. Must satisfy 2 <= k <= n.
        estimator_type: One of "s", "sloo", "sloo_minus_one".

    Returns:
        Transformed reward tensor of same shape as input.
    """
    B = rewards.numel()
    device = rewards.device
    dtype = rewards.dtype

    estimator_fn = {
        "s": s,
        "sloo": sloo,
        "sloo_minus_one": sloo_minus_one,
    }[estimator_type]

    if B % n_samples_per_prompt != 0:
        num_groups = 1
        group_sizes = [B]
    else:
        num_groups = B // n_samples_per_prompt
        group_sizes = [n_samples_per_prompt] * num_groups

    transformed = torch.zeros(B, dtype=dtype, device=device)
    offset = 0
    for gs in group_sizes:
        group_rewards = rewards[offset: offset + gs]
        transformed[offset: offset + gs] = estimator_fn(group_rewards, k)
        offset += gs

    return transformed


def _group_sizes_from_batch_size(B: int, n_samples_per_prompt: int) -> List[int]:
    if B % n_samples_per_prompt != 0:
        return [B]
    return [n_samples_per_prompt] * (B // n_samples_per_prompt)


def grpo_transform_rewards(
    rewards: torch.Tensor,
    n_samples_per_prompt: int,
    std_normalization: bool = False,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Apply GRPO/Dr.GRPO-style reward centering per prompt group.

    Args:
        rewards: Flat tensor of rewards [B].
        n_samples_per_prompt: Number of samples per prompt.
        std_normalization: Whether to divide each group by its std (vanilla GRPO).
            When False, this matches the Dr.GRPO-style centered reward transform.
        epsilon: Numerical stability constant for std normalization.

    Returns:
        Transformed rewards of shape [B].
    """
    B = rewards.numel()
    transformed = torch.zeros_like(rewards)
    offset = 0
    for gs in _group_sizes_from_batch_size(B, n_samples_per_prompt):
        group_rewards = rewards[offset : offset + gs]
        group_rewards = group_rewards - group_rewards.mean()
        if std_normalization:
            group_std = group_rewards.std(unbiased=False)
            if torch.isfinite(group_std) and group_std > epsilon:
                group_rewards = group_rewards / (group_std + epsilon)
            else:
                group_rewards = torch.zeros_like(group_rewards)
        transformed[offset : offset + gs] = group_rewards
        offset += gs
    return transformed


def _batch_standardize(scores: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Standardize scores across the full batch to keep hybrid scales comparable."""
    centered = scores - scores.mean()
    std = centered.std(unbiased=False)
    if not torch.isfinite(std) or std <= epsilon:
        return torch.zeros_like(centered)
    return centered / (std + epsilon)


def get_hybrid_pkpo_grpo_advantages(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
    n_samples_per_prompt: int,
    k: int,
    alpha: float = 0.5,
    kl_coef: float = 0.0,
    estimator_type: str = "sloo_minus_one",
    grpo_variant: str = "dr_grpo",
) -> Tuple[List[torch.Tensor], dict]:
    """Blend PKPO and (Dr.)GRPO scalar advantages, then broadcast to token level.

    The blend is performed at the per-sample advantage level:
        hybrid = (1 - alpha) * normalized_grpo + alpha * normalized_pkpo

    We normalize both branches at the batch level before mixing so neither branch
    dominates purely due to scale mismatch.
    """
    B = len(rewards)
    device = kl[0].device if kl else torch.device("cpu")

    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    rewards = rewards.float().to(device)

    std_normalization = grpo_variant == "grpo"
    grpo_scores = grpo_transform_rewards(
        rewards,
        n_samples_per_prompt=n_samples_per_prompt,
        std_normalization=std_normalization,
    )
    pkpo_scores = pkpo_transform_rewards(
        rewards,
        n_samples_per_prompt=n_samples_per_prompt,
        k=k,
        estimator_type=estimator_type,
    )

    grpo_scores = _batch_standardize(grpo_scores)
    pkpo_scores = _batch_standardize(pkpo_scores)
    hybrid_scores = _batch_standardize((1.0 - alpha) * grpo_scores + alpha * pkpo_scores)

    stats = {
        "hybrid_alpha": float(alpha),
        "hybrid_grpo_variant": grpo_variant,
        "hybrid_pkpo_k": k,
        "hybrid_pkpo_estimator": estimator_type,
        "hybrid_grpo_mean": grpo_scores.mean().item(),
        "hybrid_grpo_std": grpo_scores.std(unbiased=False).item(),
        "hybrid_pkpo_mean": pkpo_scores.mean().item(),
        "hybrid_pkpo_std": pkpo_scores.std(unbiased=False).item(),
        "hybrid_reward_mean": hybrid_scores.mean().item(),
        "hybrid_reward_std": hybrid_scores.std(unbiased=False).item(),
        "hybrid_reward_max": hybrid_scores.max().item(),
        "hybrid_reward_min": hybrid_scores.min().item(),
    }

    token_advantages = []
    for i in range(B):
        adv_scalar = hybrid_scores[i]
        token_adv = torch.ones_like(kl[i]) * adv_scalar
        if kl_coef > 0:
            token_adv = token_adv - kl_coef * kl[i]
        token_advantages.append(token_adv)

    return token_advantages, stats


def get_pkpo_advantages(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
    n_samples_per_prompt: int,
    k: int,
    kl_coef: float = 0.0,
    estimator_type: str = "sloo_minus_one",
) -> Tuple[List[torch.Tensor], dict]:
    """Compute PKPO advantages and broadcast to token level.

    Groups rewards by n_samples_per_prompt, applies the PKPO transformation,
    then broadcasts per-sample advantages to all tokens.

    Args:
        rewards: Tensor of all rewards in the batch [B].
        kl: List of per-token KL tensors, one per sample.
        n_samples_per_prompt: Number of samples per prompt (n).
        k: The k in pass@k to optimize.
        kl_coef: KL penalty coefficient.
        estimator_type: PKPO estimator variant.

    Returns:
        Tuple of (list of per-token advantage tensors, stats dict).
    """
    B = len(rewards)
    device = kl[0].device if kl else torch.device("cpu")

    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    rewards = rewards.float().to(device)

    transformed = pkpo_transform_rewards(
        rewards, n_samples_per_prompt, k, estimator_type
    )

    stats = {
        "pkpo_k": k,
        "pkpo_estimator": estimator_type,
        "pkpo_reward_mean": transformed.mean().item(),
        "pkpo_reward_std": transformed.std().item(),
        "pkpo_reward_max": transformed.max().item(),
        "pkpo_reward_min": transformed.min().item(),
    }

    token_advantages = []
    for i in range(B):
        adv_scalar = transformed[i]
        token_adv = torch.ones_like(kl[i]) * adv_scalar
        if kl_coef > 0:
            token_adv = token_adv - kl_coef * kl[i]
        token_advantages.append(token_adv)

    return token_advantages, stats
