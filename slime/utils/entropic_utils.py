"""
Entropic objective utilities for TTT-Discover.

Implements the adaptive entropic objective from the TTT-Discover paper:

    J_beta(theta) = log E_{a ~ pi_theta}[ exp(beta * R(s,a)) ]

The gradient yields importance weights:

    w_beta(a) = exp(beta * R(a)) / E[exp(beta * R(a))]

with mean-baselined advantage A(a) = w_beta(a) - 1.

Beta is set adaptively per group by solving:

    KL(q_beta || uniform) = gamma

via bisection, where q_beta(n) = exp(beta * r_n) / sum_m exp(beta * r_m).
"""

import math
from typing import List, Optional, Tuple

import torch


def compute_adaptive_beta(
    rewards: torch.Tensor,
    gamma: float = math.log(2),
    max_iterations: int = 50,
    tol: float = 1e-6,
    beta_low: float = 0.0,
    beta_high: float = 1000.0,
) -> float:
    """
    Compute adaptive beta for a group of rewards via bisection.

    Finds beta such that KL(q_beta || uniform) = gamma, where:
        q_beta(n) = exp(beta * r_n) / sum_m exp(beta * r_m)

    Args:
        rewards: Tensor of scalar rewards for the group [N].
        gamma: KL divergence constraint (default: ln(2)).
        max_iterations: Maximum bisection iterations.
        tol: Convergence tolerance on KL.
        beta_low: Lower bound for bisection.
        beta_high: Upper bound for bisection.

    Returns:
        Adaptive beta value.
    """
    N = rewards.numel()
    if N <= 1:
        return 0.0

    rewards = rewards.float()
    log_N = math.log(N)

    def kl_at_beta(beta: float) -> float:
        """KL(q_beta || uniform) where uniform = 1/N."""
        if beta == 0.0:
            return 0.0
        # Use log-sum-exp for numerical stability
        scaled = beta * rewards
        log_Z = torch.logsumexp(scaled, dim=0)
        log_q = scaled - log_Z  # log q_beta(n)
        # KL = sum q * (log q - log u) = sum q * (log q + log N)
        q = torch.exp(log_q)
        kl = (q * (log_q + log_N)).sum().item()
        return kl

    # Check if beta_high is large enough
    kl_high = kl_at_beta(beta_high)
    if kl_high < gamma:
        # Even the max beta doesn't reach the constraint; use beta_high
        return beta_high

    # Bisection
    for _ in range(max_iterations):
        beta_mid = (beta_low + beta_high) / 2.0
        kl_mid = kl_at_beta(beta_mid)

        if abs(kl_mid - gamma) < tol:
            return beta_mid

        if kl_mid < gamma:
            beta_low = beta_mid
        else:
            beta_high = beta_mid

    return (beta_low + beta_high) / 2.0


def compute_entropic_advantages(
    rewards: torch.Tensor,
    beta: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute leave-one-out (LOO) entropic advantages.

    For each sample n in a group of N samples:
        Z_{-n} = (1/(N-1)) * sum_{m != n} exp(beta * (r_m - r_max))
        A_n = exp(beta * (r_n - r_max)) / (Z_{-n} + epsilon) - 1

    Args:
        rewards: Tensor of scalar rewards [N].
        beta: Temperature parameter.
        epsilon: Small constant for numerical stability.

    Returns:
        Tensor of advantages [N].
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    rewards = rewards.float()
    r_max = rewards.max()
    shifted = beta * (rewards - r_max)  # For numerical stability

    exp_shifted = torch.exp(shifted)

    # Z_total = sum of all exp(beta * (r_m - r_max))
    Z_total = exp_shifted.sum()

    # Z_{-n} = (Z_total - exp_shifted[n]) / (N - 1)
    Z_loo = (Z_total - exp_shifted) / (N - 1)

    # A_n = exp_shifted[n] / (Z_loo + epsilon) - 1
    advantages = exp_shifted / (Z_loo + epsilon) - 1.0

    return advantages


def get_entropic_returns(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
    n_samples_per_prompt: int,
    kl_coef: float = 0.0,
    gamma_kl: float = math.log(2),
) -> Tuple[List[torch.Tensor], dict]:
    """
    Compute entropic advantages and broadcast to token level.

    Groups rewards by n_samples_per_prompt, computes adaptive beta
    and LOO entropic advantages for each group, then broadcasts
    the per-sample advantage to all tokens.

    Optionally includes a KL penalty term:
        A(a; s) = w_beta(a) - 1 - kl_coef * sum_t KL_t(a)

    Args:
        rewards: Tensor of all rewards in the batch [B].
        kl: List of per-token KL tensors, one per sample [B].
        n_samples_per_prompt: Number of samples per group (prompt).
        kl_coef: KL penalty coefficient (lambda in the paper).
        gamma_kl: KL constraint for adaptive beta (gamma in the paper).

    Returns:
        Tuple of:
            - List of per-token advantage tensors [B].
            - Dict of logged statistics.
    """
    B = len(rewards)
    device = kl[0].device if kl else torch.device("cpu")

    # Ensure rewards is a tensor
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    rewards = rewards.float().to(device)

    # Group rewards
    if B % n_samples_per_prompt != 0:
        # Fallback: treat all samples as one group
        num_groups = 1
        group_sizes = [B]
    else:
        num_groups = B // n_samples_per_prompt
        group_sizes = [n_samples_per_prompt] * num_groups

    # Compute per-sample advantages
    all_advantages = torch.zeros(B, dtype=torch.float32, device=device)
    stats = {"betas": [], "group_reward_range": []}

    offset = 0
    for g in range(num_groups):
        gs = group_sizes[g]
        group_rewards = rewards[offset : offset + gs]

        # Compute adaptive beta
        beta = compute_adaptive_beta(group_rewards, gamma=gamma_kl)
        stats["betas"].append(beta)
        stats["group_reward_range"].append(
            (group_rewards.min().item(), group_rewards.max().item())
        )

        # Compute LOO entropic advantages
        group_advantages = compute_entropic_advantages(group_rewards, beta)

        all_advantages[offset : offset + gs] = group_advantages
        offset += gs

    # Broadcast to token level and apply KL penalty
    token_advantages = []
    for i in range(B):
        adv_scalar = all_advantages[i]
        token_adv = torch.ones_like(kl[i]) * adv_scalar

        if kl_coef > 0:
            token_adv = token_adv - kl_coef * kl[i]

        token_advantages.append(token_adv)

    return token_advantages, stats
