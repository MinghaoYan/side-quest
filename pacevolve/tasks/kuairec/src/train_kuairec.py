"""Fixed-budget KuaRec training script with a FuXi-Linear-aligned model block."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 42
MAX_HISTORY = 1024
NUM_EPOCHS = 16
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
NUM_NEGATIVES = 128
MAX_WALL_TIME_SECONDS = 2400.0
TEMPERATURE = 0.05


def configure_csv_field_limit() -> None:
    """Raise the stdlib CSV field-size cap for long sequence columns."""
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            return
        except OverflowError:
            field_limit //= 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_sequence(text: str) -> list[int]:
    raw = str(text).strip()
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
    except Exception:
        value = None
    if isinstance(value, (list, tuple)):
        return [int(float(x)) for x in value]
    if value is not None:
        return [int(float(value))]
    return [int(float(part)) for part in raw.split(",") if part]


def right_pad(sequence: list[int], target_len: int) -> list[int]:
    if len(sequence) >= target_len:
        return sequence[-target_len:]
    return sequence + [0] * (target_len - len(sequence))


@dataclass
class PreparedSplit:
    history_ids: Tensor
    history_timestamps: Tensor
    history_lengths: Tensor
    target_ids: Tensor
    target_timestamps: Tensor


@dataclass
class PreparedData:
    train: PreparedSplit
    eval: PreparedSplit
    num_items: int


class SequenceDataset(Dataset):
    def __init__(self, split: PreparedSplit) -> None:
        self.history_ids = split.history_ids
        self.history_timestamps = split.history_timestamps
        self.history_lengths = split.history_lengths
        self.target_ids = split.target_ids
        self.target_timestamps = split.target_timestamps

    def __len__(self) -> int:
        return int(self.target_ids.numel())

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "history_ids": self.history_ids[index],
            "history_timestamps": self.history_timestamps[index],
            "history_lengths": self.history_lengths[index],
            "target_ids": self.target_ids[index],
            "target_timestamps": self.target_timestamps[index],
        }


def _build_split(
    sequences: list[list[int]],
    timestamps: list[list[int]],
    drop_last_events: int,
) -> PreparedSplit:
    history_ids = []
    history_timestamps = []
    history_lengths = []
    target_ids = []
    target_timestamps = []

    for seq_ids, seq_ts in zip(sequences, timestamps):
        if len(seq_ids) - drop_last_events < 2:
            continue
        usable_ids = seq_ids[: len(seq_ids) - drop_last_events]
        usable_ts = seq_ts[: len(seq_ts) - drop_last_events]
        if len(usable_ids) < 2:
            continue

        target_id = usable_ids[-1]
        target_ts = usable_ts[-1]
        history_id = usable_ids[:-1]
        history_ts = usable_ts[:-1]
        history_len = min(len(history_id), MAX_HISTORY)

        history_ids.append(right_pad(history_id, MAX_HISTORY))
        history_timestamps.append(right_pad(history_ts, MAX_HISTORY))
        history_lengths.append(history_len)
        target_ids.append(target_id)
        target_timestamps.append(target_ts)

    return PreparedSplit(
        history_ids=torch.tensor(history_ids, dtype=torch.long),
        history_timestamps=torch.tensor(history_timestamps, dtype=torch.long),
        history_lengths=torch.tensor(history_lengths, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        target_timestamps=torch.tensor(target_timestamps, dtype=torch.long),
    )


def load_or_prepare_data(dataset_csv: str) -> PreparedData:
    cache_path = f"{dataset_csv}.kuairec_cache_v2.pt"
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu")
        return PreparedData(
            train=PreparedSplit(**cached["train"]),
            eval=PreparedSplit(**cached["eval"]),
            num_items=int(cached["num_items"]),
        )

    sequences: list[list[int]] = []
    timestamps: list[list[int]] = []
    num_items = 0

    configure_csv_field_limit()
    with open(dataset_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_ids = [item + 1 for item in parse_sequence(row["sequence_item_ids"])]
            seq_ts = parse_sequence(row["sequence_timestamps"])
            if len(seq_ids) != len(seq_ts) or len(seq_ids) < 5:
                continue
            num_items = max(num_items, max(seq_ids))
            sequences.append(seq_ids)
            timestamps.append(seq_ts)

    prepared = PreparedData(
        train=_build_split(sequences, timestamps, drop_last_events=1),
        eval=_build_split(sequences, timestamps, drop_last_events=0),
        num_items=num_items,
    )
    torch.save(
        {
            "train": {
                "history_ids": prepared.train.history_ids,
                "history_timestamps": prepared.train.history_timestamps,
                "history_lengths": prepared.train.history_lengths,
                "target_ids": prepared.train.target_ids,
                "target_timestamps": prepared.train.target_timestamps,
            },
            "eval": {
                "history_ids": prepared.eval.history_ids,
                "history_timestamps": prepared.eval.history_timestamps,
                "history_lengths": prepared.eval.history_lengths,
                "target_ids": prepared.eval.target_ids,
                "target_timestamps": prepared.eval.target_timestamps,
            },
            "num_items": prepared.num_items,
        },
        cache_path,
    )
    return prepared


class BaseSequenceRecommender(nn.Module):
    def encode_users(
        self,
        history_ids: Tensor,
        history_timestamps: Tensor,
        history_lengths: Tensor,
        target_timestamps: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def score_all_items(self, user_embeddings: Tensor) -> Tensor:
        raise NotImplementedError

    def score_candidates(self, user_embeddings: Tensor, candidate_ids: Tensor) -> Tensor:
        all_scores = self.score_all_items(user_embeddings)
        return all_scores.gather(1, candidate_ids - 1)


def ensure_model_contract(model: nn.Module) -> None:
    required_methods = ("encode_users", "score_all_items")
    missing = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing:
        missing_str = ", ".join(missing)
        raise TypeError(f"Model returned by build_model(...) is missing required methods: {missing_str}")


def score_training_candidates(
    model: nn.Module,
    user_embeddings: Tensor,
    candidate_ids: Tensor,
) -> Tensor:
    score_candidates = getattr(model, "score_candidates", None)
    if callable(score_candidates):
        return score_candidates(user_embeddings, candidate_ids)
    all_scores = model.score_all_items(user_embeddings)
    return all_scores.gather(1, candidate_ids - 1)


# RegexTagCustomPruningAlgorithmStart

MODEL_DIM = 128
NUM_BLOCKS = 4
NUM_CONTENT_HEADS = 4
CONTENT_HEAD_DIM = 32
CONTENT_VALUE_DIM = 32
TEMPORAL_HEADS = 8
POSITION_DIM = 32
CHUNK_SIZE = 128
DROPOUT_RATE = 0.5
L2_NORM_EPS = 1e-6


def build_valid_mask(lengths: Tensor, max_len: int) -> Tensor:
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def rms_norm(inputs: Tensor, eps: float = 1e-6) -> Tensor:
    return F.rms_norm(inputs, normalized_shape=(inputs.size(-1),), eps=eps)


def chunkwise_recurrent_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    decay_steps: Tensor,
    mask: Tensor,
    chunk_size: int,
) -> Tensor:
    batch_size, seq_len, num_heads, q_dim = q.shape
    value_dim = v.size(-1)
    dtype = q.dtype

    mask_4d = mask.unsqueeze(-1).unsqueeze(-1).to(dtype)
    q = q * mask_4d
    k = k * mask_4d
    v = v * mask_4d
    safe_decay = torch.where(
        mask.unsqueeze(-1),
        decay_steps.to(dtype),
        torch.ones_like(decay_steps, dtype=dtype),
    )

    state = torch.zeros(
        batch_size,
        num_heads,
        q_dim,
        value_dim,
        device=q.device,
        dtype=dtype,
    )
    outputs = []

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(seq_len, chunk_start + chunk_size)
        q_chunk = q[:, chunk_start:chunk_end]
        k_chunk = k[:, chunk_start:chunk_end]
        v_chunk = v[:, chunk_start:chunk_end]
        decay_chunk = safe_decay[:, chunk_start:chunk_end]
        chunk_mask = mask[:, chunk_start:chunk_end]
        chunk_len = q_chunk.size(1)

        prefix = torch.cumprod(decay_chunk.clamp_min(1e-8), dim=1)
        prev_query = q_chunk * prefix.unsqueeze(-1)
        prev_output = torch.einsum("bchd,bhde->bche", prev_query, state)

        decay_matrix = prefix.unsqueeze(2) / prefix.unsqueeze(1).clamp_min(1e-8)
        decay_matrix = decay_matrix.permute(0, 3, 1, 2)
        causal = torch.tril(
            torch.ones(chunk_len, chunk_len, device=q.device, dtype=dtype)
        )
        attn_scores = torch.einsum("bchd,bmhd->bhcm", q_chunk, k_chunk)
        weights = attn_scores * decay_matrix * causal
        intra_output = torch.einsum("bhcm,bmhe->bche", weights, v_chunk)

        chunk_output = (prev_output + intra_output) * chunk_mask.unsqueeze(-1).unsqueeze(-1).to(dtype)
        outputs.append(chunk_output)

        prefix_last = prefix[:, -1, :].unsqueeze(-1).unsqueeze(-1)
        state_weights = (prefix[:, -1:, :] / prefix.clamp_min(1e-8)).squeeze(1)
        state = state * prefix_last + torch.einsum(
            "bchd,bche->bhde",
            k_chunk * state_weights.unsqueeze(-1),
            v_chunk,
        )

    return torch.cat(outputs, dim=1)


class RetentionChannel(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, value_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.chunk_size = chunk_size
        self.gamma_param = nn.Parameter(torch.empty(num_heads).normal_(mean=0.0, std=0.02))

    def decay(self) -> Tensor:
        return torch.exp(-torch.cumsum(F.softplus(self.gamma_param), dim=0))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        gamma = self.decay().to(q.dtype)
        decay_steps = gamma.view(1, 1, self.num_heads).expand(q.size(0), q.size(1), -1)
        outputs = chunkwise_recurrent_attention(
            q=q,
            k=k,
            v=v,
            decay_steps=decay_steps,
            mask=mask,
            chunk_size=self.chunk_size,
        )
        return outputs.reshape(q.size(0), q.size(1), self.num_heads * self.value_dim)


class LinearTemporalChannel(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        base: int,
        start_index: int,
        base_stride: int,
        chunk_size: int,
        learnable_gamma: bool,
        use_augment_connection: bool,
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_channels = num_heads * 2
        self.head_dim = dim_model // self.num_channels
        self.chunk_size = chunk_size
        self.use_augment_connection = use_augment_connection

        scales = torch.arange(
            start_index,
            start_index + num_heads * base_stride,
            base_stride,
            dtype=torch.float32,
        )
        self.register_buffer("intervals", torch.pow(float(base), scales).to(torch.long))
        self.register_buffer("scale_factor", 2.0 * torch.pi * torch.pow(1.0 / float(base), scales))
        self.gamma_param = nn.Parameter(
            torch.zeros(num_heads), requires_grad=learnable_gamma
        )
        self.proj_v = nn.Linear(dim_model, dim_model, bias=False)
        if use_augment_connection:
            self.alpha = nn.Parameter(torch.empty(self.num_channels).normal_(mean=0.0, std=0.02))
            self.beta = nn.Parameter(torch.ones(self.num_channels))

    def _pow_decay(self, gamma: Tensor, scaled_delta: Tensor) -> Tensor:
        return torch.exp(torch.log(gamma.clamp_min(1e-6)) * scaled_delta)

    def _build_query_key(
        self,
        timestamps: Tensor,
        next_timestamps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        float_timestamps = timestamps.to(torch.float32)
        float_next = next_timestamps.to(torch.float32)
        intervals = self.intervals.to(device=timestamps.device)
        scale_factor = self.scale_factor.to(device=timestamps.device)

        theta_k = torch.remainder(float_timestamps.unsqueeze(-1), intervals.unsqueeze(0).unsqueeze(0).to(torch.float32))
        theta_k = theta_k * scale_factor.unsqueeze(0).unsqueeze(0)
        theta_q = torch.remainder(float_next.unsqueeze(-1), intervals.unsqueeze(0).unsqueeze(0).to(torch.float32))
        theta_q = theta_q * scale_factor.unsqueeze(0).unsqueeze(0)

        cos_t = torch.cos(theta_k)
        sin_t = torch.sin(theta_k)
        key = torch.stack((cos_t, sin_t), dim=-1).repeat(1, 1, 2, 1)

        cos_q = torch.cos(theta_q)
        sin_q = torch.sin(theta_q)
        query_sin = torch.stack((sin_q, -cos_q), dim=-1)
        query_cos = torch.stack((cos_q, sin_q), dim=-1)
        query = torch.cat((query_sin, query_cos), dim=2)

        gamma = torch.sigmoid(self.gamma_param).to(float_timestamps.dtype)
        delta_to_target = (float_next - float_timestamps).clamp_min(0.0)
        scaled_delta = delta_to_target.unsqueeze(-1) * scale_factor.unsqueeze(0).unsqueeze(0)
        query_decay = self._pow_decay(gamma.unsqueeze(0).unsqueeze(0), scaled_delta)
        query = query * query_decay.repeat(1, 1, 2).unsqueeze(-1)

        step_delta = torch.zeros_like(float_timestamps)
        step_delta[:, 1:] = (float_timestamps[:, 1:] - float_timestamps[:, :-1]).clamp_min(0.0)
        scaled_steps = step_delta.unsqueeze(-1) * scale_factor.unsqueeze(0).unsqueeze(0)
        decay_steps = self._pow_decay(gamma.unsqueeze(0).unsqueeze(0), scaled_steps).repeat(1, 1, 2)
        return query, key, decay_steps

    def forward(
        self,
        inputs: Tensor,
        timestamps: Tensor,
        next_timestamps: Tensor,
        mask: Tensor,
    ) -> Tensor:
        values = self.proj_v(inputs).view(inputs.size(0), inputs.size(1), self.num_channels, self.head_dim)
        query, key, decay_steps = self._build_query_key(timestamps, next_timestamps)
        outputs = chunkwise_recurrent_attention(
            q=query.to(values.dtype),
            k=key.to(values.dtype),
            v=values,
            decay_steps=decay_steps.to(values.dtype),
            mask=mask,
            chunk_size=self.chunk_size,
        )
        if self.use_augment_connection:
            outputs = outputs * self.alpha.view(1, 1, -1, 1).to(outputs.dtype)
            outputs = outputs + values * self.beta.view(1, 1, -1, 1).to(values.dtype)
        return outputs.reshape(inputs.size(0), inputs.size(1), self.dim_model)


class LinearPositionalChannel(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        dim_model: int,
        embedding_dim: int,
        chunk_size: int,
        aug_current: bool,
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.chunk_size = chunk_size
        self.position_dim = embedding_dim
        self.proj_v = nn.Linear(dim_model, dim_model, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(max_seq_len, embedding_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.aug_current = aug_current
        if aug_current:
            self.alpha = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.02))
            self.beta = nn.Parameter(torch.ones(1))

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        batch_size, seq_len, _ = inputs.shape
        values = self.proj_v(inputs).unsqueeze(2)
        query = self.position_embedding[:seq_len].view(1, seq_len, 1, self.position_dim)
        scale = float(max(self.position_dim // 2, 1))
        key = query / scale
        decay_steps = torch.ones(batch_size, seq_len, 1, device=inputs.device, dtype=inputs.dtype)
        outputs = chunkwise_recurrent_attention(
            q=query.expand(batch_size, -1, -1, -1).to(inputs.dtype),
            k=key.expand(batch_size, -1, -1, -1).to(inputs.dtype),
            v=values,
            decay_steps=decay_steps,
            mask=mask,
            chunk_size=self.chunk_size,
        ).squeeze(2)
        if self.aug_current:
            outputs = outputs * self.alpha.to(outputs.dtype) + self.beta.to(outputs.dtype) * values.squeeze(2)
        return outputs


class MultiStageFeedForward(nn.Module):
    def __init__(self, attn_dim: int, dim_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lin0 = nn.Linear(attn_dim, dim_model, bias=False)
        self.lin1 = nn.Linear(dim_model, dim_model, bias=False)
        self.lin2 = nn.Linear(dim_model, dim_model, bias=False)
        self.lin3 = nn.Linear(dim_model, dim_model, bias=False)

    def forward(self, attn_output: Tensor, residual: Tensor) -> Tensor:
        outputs = self.lin0(self.dropout(attn_output)) + residual
        normed = self.dropout(rms_norm(outputs))
        feedforward = F.silu(self.lin1(normed)) * self.lin3(normed)
        return self.lin2(feedforward) + outputs


class FuXiLinearBlock(nn.Module):
    def __init__(self, max_seq_len: int, learnable_temporal_gamma: bool) -> None:
        super().__init__()
        attn_dim = MODEL_DIM * 3
        qkv_dim = NUM_CONTENT_HEADS * CONTENT_HEAD_DIM
        value_dim = NUM_CONTENT_HEADS * CONTENT_VALUE_DIM
        self.input_norm = nn.LayerNorm(MODEL_DIM)
        self.uvqk = nn.Linear(MODEL_DIM, attn_dim + qkv_dim + qkv_dim + value_dim, bias=False)
        self.retention_channel = RetentionChannel(
            num_heads=NUM_CONTENT_HEADS,
            head_dim=CONTENT_HEAD_DIM,
            value_dim=CONTENT_VALUE_DIM,
            chunk_size=CHUNK_SIZE,
        )
        self.position_channel = LinearPositionalChannel(
            max_seq_len=max_seq_len,
            dim_model=MODEL_DIM,
            embedding_dim=POSITION_DIM,
            chunk_size=CHUNK_SIZE,
            aug_current=True,
        )
        self.temporal_channel = LinearTemporalChannel(
            dim_model=MODEL_DIM,
            num_heads=TEMPORAL_HEADS,
            base=2,
            start_index=10,
            base_stride=3,
            chunk_size=CHUNK_SIZE,
            learnable_gamma=learnable_temporal_gamma,
            use_augment_connection=True,
        )
        self.ffn = MultiStageFeedForward(attn_dim=attn_dim, dim_model=MODEL_DIM, dropout=DROPOUT_RATE)

    def forward(
        self,
        inputs: Tensor,
        timestamps: Tensor,
        next_timestamps: Tensor,
        mask: Tensor,
    ) -> Tensor:
        batch_size, seq_len, _ = inputs.shape
        normed = self.input_norm(inputs)
        projected = F.silu(self.uvqk(normed))
        gate_values, query, key, value = torch.split(
            projected,
            [MODEL_DIM * 3, MODEL_DIM, MODEL_DIM, MODEL_DIM],
            dim=-1,
        )

        query = query.view(batch_size, seq_len, NUM_CONTENT_HEADS, CONTENT_HEAD_DIM)
        key = key.view(batch_size, seq_len, NUM_CONTENT_HEADS, CONTENT_HEAD_DIM)
        value = value.view(batch_size, seq_len, NUM_CONTENT_HEADS, CONTENT_VALUE_DIM)

        retention_output = self.retention_channel(query, key, value, mask)
        positional_output = self.position_channel(normed, mask)
        temporal_output = self.temporal_channel(normed, timestamps, next_timestamps, mask)

        combined = torch.cat(
            (
                F.layer_norm(retention_output, normalized_shape=(MODEL_DIM,)),
                F.layer_norm(positional_output, normalized_shape=(MODEL_DIM,)),
                F.layer_norm(temporal_output, normalized_shape=(MODEL_DIM,)),
            ),
            dim=-1,
        )
        attn_output = gate_values * combined
        outputs = self.ffn(attn_output, inputs)
        return outputs * mask.unsqueeze(-1).to(outputs.dtype)


class FuXiLinearSequenceModel(BaseSequenceRecommender):
    def __init__(self, num_items: int, max_history: int) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_history = max_history
        self.item_embedding = nn.Embedding(num_items + 1, MODEL_DIM, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history, MODEL_DIM)
        self.input_dropout = nn.Dropout(DROPOUT_RATE)
        self.blocks = nn.ModuleList(
            [
                FuXiLinearBlock(
                    max_seq_len=max_history,
                    learnable_temporal_gamma=(block_idx == 0),
                )
                for block_idx in range(NUM_BLOCKS)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        if self.item_embedding.padding_idx is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.item_embedding.padding_idx].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def _build_next_timestamps(
        self,
        history_timestamps: Tensor,
        history_lengths: Tensor,
        target_timestamps: Tensor,
    ) -> Tensor:
        next_timestamps = torch.zeros_like(history_timestamps)
        next_timestamps[:, :-1] = history_timestamps[:, 1:]
        batch_indices = torch.arange(history_lengths.size(0), device=history_lengths.device)
        last_indices = history_lengths.clamp_min(1) - 1
        next_timestamps[batch_indices, last_indices] = target_timestamps
        mask = build_valid_mask(history_lengths, history_timestamps.size(1))
        return next_timestamps * mask.to(next_timestamps.dtype)

    def encode_users(
        self,
        history_ids: Tensor,
        history_timestamps: Tensor,
        history_lengths: Tensor,
        target_timestamps: Tensor,
    ) -> Tensor:
        mask = build_valid_mask(history_lengths, history_ids.size(1))
        positions = torch.arange(self.max_history, device=history_ids.device).unsqueeze(0)
        token_embeddings = self.item_embedding(history_ids) * math.sqrt(MODEL_DIM)
        token_embeddings = token_embeddings + self.position_embedding(positions)
        token_embeddings = self.input_dropout(token_embeddings)
        token_embeddings = token_embeddings * mask.unsqueeze(-1).to(token_embeddings.dtype)

        next_timestamps = self._build_next_timestamps(
            history_timestamps=history_timestamps,
            history_lengths=history_lengths,
            target_timestamps=target_timestamps,
        )

        outputs = token_embeddings
        for block in self.blocks:
            outputs = block(outputs, history_timestamps, next_timestamps, mask)
        outputs = F.normalize(outputs, dim=-1, eps=L2_NORM_EPS)

        batch_indices = torch.arange(history_lengths.size(0), device=history_ids.device)
        last_indices = history_lengths.clamp_min(1) - 1
        return outputs[batch_indices, last_indices]

    def score_all_items(self, user_embeddings: Tensor) -> Tensor:
        item_embeddings = self.item_embedding.weight[1 : self.num_items + 1]
        item_embeddings = F.normalize(item_embeddings, dim=-1, eps=L2_NORM_EPS)
        return user_embeddings @ item_embeddings.t() / TEMPERATURE

    def score_candidates(self, user_embeddings: Tensor, candidate_ids: Tensor) -> Tensor:
        item_embeddings = self.item_embedding(candidate_ids)
        item_embeddings = F.normalize(item_embeddings, dim=-1, eps=L2_NORM_EPS)
        return (user_embeddings.unsqueeze(1) * item_embeddings).sum(dim=-1) / TEMPERATURE


def build_model(num_items: int, max_history: int) -> BaseSequenceRecommender:
    return FuXiLinearSequenceModel(num_items=num_items, max_history=max_history)


# RegexTagCustomPruningAlgorithmEnd


def run_syntax_only() -> None:
    model = build_model(num_items=256, max_history=MAX_HISTORY)
    ensure_model_contract(model)

    history_ids = torch.randint(1, 128, (4, MAX_HISTORY))
    history_lengths = torch.full((4,), MAX_HISTORY // 2, dtype=torch.long)
    valid_len = int(history_lengths[0].item())
    history_ids[:, valid_len:] = 0
    history_timestamps = torch.zeros((4, MAX_HISTORY), dtype=torch.long)
    history_timestamps[:, :valid_len] = torch.arange(valid_len, dtype=torch.long)
    target_timestamps = torch.full((4,), valid_len + 1, dtype=torch.long)
    user_embeddings = model.encode_users(history_ids, history_timestamps, history_lengths, target_timestamps)
    candidate_ids = torch.randint(1, 128, (4, 16))
    logits = score_training_candidates(model, user_embeddings, candidate_ids)
    assert user_embeddings.ndim == 2
    assert logits.shape == (4, 16)
    print("syntax_ok")


def sample_negatives(target_ids: Tensor, num_items: int, num_negatives: int) -> Tensor:
    negatives = torch.randint(
        1,
        num_items,
        (target_ids.size(0), num_negatives),
        device=target_ids.device,
    )
    negatives = negatives + (negatives >= target_ids.unsqueeze(1)).long()
    return negatives


def build_loader(split: PreparedSplit, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SequenceDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def all_finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def invalid_metrics(reason: str) -> dict[str, float | bool | str]:
    return {
        "combined_score": 0.0,
        "ndcg@10": 0.0,
        "ndcg@50": 0.0,
        "hr@10": 0.0,
        "hr@50": 0.0,
        "mrr": 0.0,
        "valid_run": False,
        "failure_reason": reason,
    }


def evaluate_model(
    model: BaseSequenceRecommender,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | bool | str]:
    model.eval()
    ndcg10_sum = 0.0
    ndcg50_sum = 0.0
    hr10_sum = 0.0
    hr50_sum = 0.0
    mrr_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            history_ids = batch["history_ids"].to(device)
            history_timestamps = batch["history_timestamps"].to(device)
            history_lengths = batch["history_lengths"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_timestamps = batch["target_timestamps"].to(device)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                user_embeddings = model.encode_users(
                    history_ids,
                    history_timestamps,
                    history_lengths,
                    target_timestamps,
                )
                scores = model.score_all_items(user_embeddings).float()
            if not all_finite(user_embeddings):
                return invalid_metrics("Non-finite user embeddings encountered during evaluation.")
            if not all_finite(scores):
                return invalid_metrics("Non-finite item scores encountered during evaluation.")

            for row_idx in range(scores.size(0)):
                length = int(history_lengths[row_idx].item())
                seen = history_ids[row_idx, :length]
                seen = seen[seen > 0] - 1
                seen = seen[seen != (target_ids[row_idx] - 1)]
                if seen.numel() > 0:
                    scores[row_idx, seen] = -1e9

            target_index = target_ids - 1
            target_scores = scores.gather(1, target_index.unsqueeze(1)).squeeze(1)
            if not all_finite(target_scores):
                return invalid_metrics("Non-finite target scores encountered during evaluation.")

            ranks = 1 + (scores > target_scores.unsqueeze(1)).sum(dim=1)
            ranks_f = ranks.to(torch.float32)
            count += int(target_ids.size(0))

            mrr_sum += (1.0 / ranks_f).sum().item()

            hit10 = ranks <= 10
            hit50 = ranks <= 50
            hr10_sum += hit10.to(torch.float32).sum().item()
            hr50_sum += hit50.to(torch.float32).sum().item()

            ndcg10_sum += torch.where(
                hit10,
                1.0 / torch.log2(ranks_f + 1.0),
                torch.zeros_like(ranks_f),
            ).sum().item()
            ndcg50_sum += torch.where(
                hit50,
                1.0 / torch.log2(ranks_f + 1.0),
                torch.zeros_like(ranks_f),
            ).sum().item()

    ndcg10 = ndcg10_sum / max(count, 1)
    ndcg50 = ndcg50_sum / max(count, 1)
    hr10 = hr10_sum / max(count, 1)
    hr50 = hr50_sum / max(count, 1)
    mrr = mrr_sum / max(count, 1)
    combined = (ndcg10 + hr10 + mrr) / 3.0
    return {
        "combined_score": combined,
        "ndcg@10": ndcg10,
        "ndcg@50": ndcg50,
        "hr@10": hr10,
        "hr@50": hr50,
        "mrr": mrr,
        "valid_run": True,
    }


def save_analysis_artifacts(
    model: nn.Module,
    prepared: PreparedData,
    metrics: dict[str, float | bool | str],
) -> None:
    artifact_dir = os.environ.get("PACEVOLVE_ARTIFACT_DIR", "").strip()
    if not artifact_dir:
        return
    os.makedirs(artifact_dir, exist_ok=True)

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    checkpoint_path = os.path.join(artifact_dir, "final_model.pt")
    metadata_path = os.path.join(artifact_dir, "analysis_artifact.json")

    torch.save(
        {
            "state_dict": state_dict,
            "model_class": model.__class__.__name__,
            "num_items": prepared.num_items,
            "max_history": MAX_HISTORY,
            "metrics": metrics,
            "train_config": {
                "seed": SEED,
                "num_epochs": NUM_EPOCHS,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "num_negatives": NUM_NEGATIVES,
                "temperature": TEMPERATURE,
            },
        },
        checkpoint_path,
    )

    metadata = {
        "checkpoint_path": checkpoint_path,
        "model_class": model.__class__.__name__,
        "num_items": prepared.num_items,
        "max_history": MAX_HISTORY,
        "param_count": int(sum(param.numel() for param in model.parameters())),
        "metrics": metrics,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def train_and_evaluate(dataset_csv: str) -> None:
    set_seed(SEED)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    wall_start = time.time()
    prepared = load_or_prepare_data(dataset_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_items=prepared.num_items, max_history=MAX_HISTORY).to(device)
    ensure_model_contract(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = build_loader(prepared.train, TRAIN_BATCH_SIZE, shuffle=True)
    eval_loader = build_loader(prepared.eval, EVAL_BATCH_SIZE, shuffle=False)

    last_loss = 0.0
    valid_run = True
    failure_reason = ""
    train_start = time.time()
    for _epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            history_ids = batch["history_ids"].to(device)
            history_timestamps = batch["history_timestamps"].to(device)
            history_lengths = batch["history_lengths"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_timestamps = batch["target_timestamps"].to(device)

            negative_ids = sample_negatives(target_ids, prepared.num_items, NUM_NEGATIVES)
            candidate_ids = torch.cat((target_ids.unsqueeze(1), negative_ids), dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                user_embeddings = model.encode_users(
                    history_ids,
                    history_timestamps,
                    history_lengths,
                    target_timestamps,
                )
                logits = score_training_candidates(model, user_embeddings, candidate_ids)
                loss = F.cross_entropy(
                    logits,
                    torch.zeros(logits.size(0), dtype=torch.long, device=device),
                )
            if not all_finite(user_embeddings):
                valid_run = False
                failure_reason = "Non-finite user embeddings encountered during training."
                break
            if not all_finite(logits):
                valid_run = False
                failure_reason = "Non-finite logits encountered during training."
                break
            if not all_finite(loss.reshape(1)):
                valid_run = False
                failure_reason = "Non-finite training loss encountered."
                break
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        if not valid_run:
            break
    train_time = time.time() - train_start

    eval_start = time.time()
    if valid_run:
        metrics = evaluate_model(model, eval_loader, device)
    else:
        metrics = invalid_metrics(failure_reason)
    eval_time = time.time() - eval_start
    wall_time = time.time() - wall_start
    metrics.update(
        {
            "last_train_loss": last_loss,
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "wall_time_sec": wall_time,
            "within_budget": wall_time <= MAX_WALL_TIME_SECONDS,
            "num_items": prepared.num_items,
            "num_train_users": int(prepared.train.target_ids.numel()),
            "num_eval_users": int(prepared.eval.target_ids.numel()),
        }
    )
    save_analysis_artifacts(model, prepared, metrics)
    print(f"Candidate: {json.dumps(metrics, sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the fixed-budget KuaRec FuXi-linear baseline.")
    parser.add_argument("--dataset_csv", type=str, default="")
    parser.add_argument("--syntax_only", action="store_true")
    args = parser.parse_args()

    if args.syntax_only:
        run_syntax_only()
        return

    if not args.dataset_csv:
        raise ValueError("--dataset_csv is required unless --syntax_only is set.")

    train_and_evaluate(args.dataset_csv)


if __name__ == "__main__":
    main()
