"""
Tests for PKPO (Pass-at-k Policy Optimization) utilities.

Validates our PyTorch implementation against the reference NumPy
implementation from the paper (Listing 1).
"""

import numpy as np
import torch
import pytest


# ---- Reference implementation from the paper (Listing 1, NumPy) ----

def _ref_m_normed(N, K, i, j):
    if i == j and i >= K - 1:
        return (
            K / (N - K + 1) *
            np.prod(np.arange(i - K + 2, i + 1) / np.arange(N - K + 2, N + 1))
        )
    elif j > i and j >= K - 1 and K >= 2:
        return (
            K / (N - K + 1) * (K - 1) / N *
            np.prod(np.arange(j - K + 2, j) / np.arange(N - K + 2, N))
        )
    return 0


def _ref_m_diagonal(N, K):
    return np.array([_ref_m_normed(N, K, i, i) for i in range(N)])


def ref_rho(g, K):
    return (np.sort(g) * _ref_m_diagonal(len(g), K)).sum()


def _ref_delta(N, K, i):
    return _ref_m_normed(N, K, i, i + 1) - _ref_m_normed(N, K, i + 1, i + 1)


def _ref_deltas(N, K):
    return np.array([_ref_delta(N - 1, K, i) for i in range(N - 2)])


def _ref_sorted_apply(func):
    def inner(x, *args, **kwargs):
        i_sort = np.argsort(x)
        func_x = np.zeros_like(x)
        func_x[i_sort] = func(x[i_sort], *args, **kwargs)
        return func_x
    return inner


@_ref_sorted_apply
def ref_s(g, K):
    N = len(g)
    c = g * _ref_m_diagonal(N, K)
    c[:(N - 1)] += g[1:] * _ref_deltas(N + 1, K)
    return np.cumsum(c[::-1])[::-1]


@_ref_sorted_apply
def ref_b(g, K):
    N = len(g)
    w = (_ref_m_diagonal(N - 1, K) * np.arange(1, N)).astype(float)
    w[1:] += _ref_deltas(N, K) * np.arange(1, N - 1)
    c1 = np.array([(w * g[1:]).sum()])
    c2 = (g[:-1] - g[1:]) * w
    return np.cumsum(np.concatenate((c1, c2)))


def ref_sloo(g, K):
    return ref_s(g, K) - ref_b(g, K) / (len(g) - 1)


def ref_sloo_minus_one(g, K):
    return ref_s(g, K) - ref_b(g, K - 1) * K / (K - 1) / len(g)


# ---- Our implementation (PyTorch) ----
# Direct import to avoid pulling in heavy dependencies (sglang, megatron)
# through slime.utils.__init__

import sys
import os
import importlib.util

_pkpo_path = os.path.join(os.path.dirname(__file__), "..", "slime", "utils", "pkpo_utils.py")
_spec = importlib.util.spec_from_file_location("pkpo_utils", _pkpo_path)
_pkpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pkpo)
rho = _pkpo.rho
s = _pkpo.s
sloo = _pkpo.sloo
sloo_minus_one = _pkpo.sloo_minus_one
pkpo_transform_rewards = _pkpo.pkpo_transform_rewards


# ---- Tests ----

def _random_rewards(n, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n).astype(np.float64)


@pytest.mark.parametrize("n,k", [(8, 2), (8, 4), (8, 8), (16, 4), (16, 8), (16, 16), (4, 2)])
class TestPKPOAgainstReference:
    """Compare our PyTorch implementation against the paper's NumPy reference."""

    def test_rho(self, n, k):
        g_np = _random_rewards(n)
        g_torch = torch.tensor(g_np)
        expected = ref_rho(g_np, k)
        actual = rho(g_torch, k)
        assert abs(expected - actual) < 1e-10, f"rho mismatch: {expected} vs {actual}"

    def test_s(self, n, k):
        g_np = _random_rewards(n)
        g_torch = torch.tensor(g_np)
        expected = ref_s(g_np.copy(), k)
        actual = s(g_torch, k).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_sloo(self, n, k):
        g_np = _random_rewards(n)
        g_torch = torch.tensor(g_np)
        expected = ref_sloo(g_np.copy(), k)
        actual = sloo(g_torch, k).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_sloo_minus_one(self, n, k):
        if k < 2:
            pytest.skip("sloo_minus_one requires k >= 2")
        g_np = _random_rewards(n)
        g_torch = torch.tensor(g_np)
        expected = ref_sloo_minus_one(g_np.copy(), k)
        actual = sloo_minus_one(g_torch, k).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-10)


class TestPKPOProperties:
    """Test mathematical properties of the PKPO estimators."""

    def test_rho_k1_equals_mean(self):
        """When k=1, maxg@1 = E[g], so rho should equal the mean."""
        g = torch.tensor([0.1, 0.5, 0.3, 0.8])
        assert abs(rho(g, 1) - g.mean().item()) < 1e-10

    def test_rho_kn_equals_max(self):
        """When k=n, maxg@n should weight only the maximum sample."""
        g = torch.tensor([0.1, 0.5, 0.3, 0.8])
        r = rho(g, len(g))
        assert abs(r - 0.8) < 1e-10

    def test_s_k1_is_uniform(self):
        """When k=1, all samples should be weighted equally."""
        g = torch.tensor([0.1, 0.5, 0.3, 0.8])
        result = s(g, 1)
        assert torch.allclose(result, result.mean() * torch.ones_like(result), atol=1e-10)

    def test_pkpo_transform_batch(self):
        """Test that pkpo_transform_rewards works on grouped batches."""
        rewards = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.7])
        n_samples = 4
        k = 2
        transformed = pkpo_transform_rewards(rewards, n_samples, k, "sloo_minus_one")
        assert transformed.shape == rewards.shape

        # Verify each group is independently transformed
        g1 = sloo_minus_one(rewards[:4], k)
        g2 = sloo_minus_one(rewards[4:], k)
        expected = torch.cat([g1, g2])
        torch.testing.assert_close(transformed.to(torch.float64), expected, atol=1e-10, rtol=1e-10)

    def test_sloo_minus_one_monotonic_in_reward(self):
        """Higher raw rewards should generally get higher transformed rewards
        within a group (at least the max should get the highest)."""
        g = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        result = sloo_minus_one(g, 4)
        assert result.argmax() == g.argmax()

    def test_different_k_values(self):
        """Verify that different k values produce different transformations."""
        g = torch.randn(16)
        r2 = sloo_minus_one(g, 2)
        r4 = sloo_minus_one(g, 4)
        r8 = sloo_minus_one(g, 8)
        assert not torch.allclose(r2, r4)
        assert not torch.allclose(r4, r8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
