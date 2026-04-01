"""Fixed-budget KuaRec training script with an editable FuXi-linear-style model block."""

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
MAX_HISTORY = 64
NUM_EPOCHS = 16
TRAIN_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 2048
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_NEGATIVES = 127
MAX_WALL_TIME_SECONDS = 300.0


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


def left_pad(sequence: list[int], target_len: int) -> list[int]:
    if len(sequence) >= target_len:
        return sequence[-target_len:]
    return [0] * (target_len - len(sequence)) + sequence


@dataclass
class PreparedSplit:
    history_ids: Tensor
    history_timestamps: Tensor
    history_lengths: Tensor
    target_ids: Tensor


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

    def __len__(self) -> int:
        return int(self.target_ids.numel())

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "history_ids": self.history_ids[index],
            "history_timestamps": self.history_timestamps[index],
            "history_lengths": self.history_lengths[index],
            "target_ids": self.target_ids[index],
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

    for seq_ids, seq_ts in zip(sequences, timestamps):
        if len(seq_ids) - drop_last_events < 2:
            continue
        usable_ids = seq_ids[: len(seq_ids) - drop_last_events]
        usable_ts = seq_ts[: len(seq_ts) - drop_last_events]
        if len(usable_ids) < 2:
            continue
        target_id = usable_ids[-1]
        history_id = usable_ids[:-1]
        history_ts = usable_ts[:-1]
        history_len = min(len(history_id), MAX_HISTORY)

        history_ids.append(left_pad(history_id, MAX_HISTORY))
        history_timestamps.append(left_pad(history_ts, MAX_HISTORY))
        history_lengths.append(history_len)
        target_ids.append(target_id)

    return PreparedSplit(
        history_ids=torch.tensor(history_ids, dtype=torch.long),
        history_timestamps=torch.tensor(history_timestamps, dtype=torch.long),
        history_lengths=torch.tensor(history_lengths, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
    )


def load_or_prepare_data(dataset_csv: str) -> PreparedData:
    cache_path = f"{dataset_csv}.kuairec_cache.pt"
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
            },
            "eval": {
                "history_ids": prepared.eval.history_ids,
                "history_timestamps": prepared.eval.history_timestamps,
                "history_lengths": prepared.eval.history_lengths,
                "target_ids": prepared.eval.target_ids,
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

class FuXiSequenceFeatures:
    def __init__(
        self,
        max_history: int,
        num_gap_buckets: int = 64,
        num_recency_buckets: int = 64,
    ) -> None:
        self.max_history = max_history
        self.num_gap_buckets = num_gap_buckets
        self.num_recency_buckets = num_recency_buckets

    def make_mask(self, history_lengths: Tensor) -> Tensor:
        positions = torch.arange(self.max_history, device=history_lengths.device)
        return positions.unsqueeze(0) >= (self.max_history - history_lengths).unsqueeze(1)

    def _bucketize_log(self, values: Tensor, num_buckets: int) -> Tensor:
        values = values.to(torch.float32).clamp_min(0.0)
        buckets = torch.floor(torch.log2(values + 1.0))
        return buckets.clamp_(0, num_buckets - 1).to(torch.long)

    def build(self, history_timestamps: Tensor, history_lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mask = self.make_mask(history_lengths)
        prev_timestamps = torch.roll(history_timestamps, shifts=1, dims=1)
        prev_mask = torch.roll(mask, shifts=1, dims=1)
        prev_mask[:, 0] = False

        gap_values = torch.where(
            mask & prev_mask,
            (history_timestamps - prev_timestamps).clamp_min(0),
            torch.zeros_like(history_timestamps),
        )
        latest_timestamps = history_timestamps[:, -1:].expand_as(history_timestamps)
        recency_values = torch.where(
            mask,
            (latest_timestamps - history_timestamps).clamp_min(0),
            torch.zeros_like(history_timestamps),
        )

        gap_buckets = self._bucketize_log(gap_values, self.num_gap_buckets)
        recency_buckets = self._bucketize_log(recency_values, self.num_recency_buckets)
        return gap_buckets, recency_buckets, mask


class FuXiLinearBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        ffn_multiplier: int,
        temporal_kernel_size: int,
        positional_kernel_size: int,
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.pre_norm = nn.LayerNorm(dim_model)
        self.input_proj = nn.Linear(dim_model, dim_model + 3 * self.inner_dim)
        self.content_proj = nn.Linear(self.inner_dim, dim_model)
        self.temporal_conv = nn.Conv1d(
            dim_model,
            dim_model,
            kernel_size=temporal_kernel_size,
            groups=dim_model,
            padding=0,
        )
        self.positional_conv = nn.Conv1d(
            dim_model,
            dim_model,
            kernel_size=positional_kernel_size,
            groups=dim_model,
            padding=0,
        )
        self.mix_proj = nn.Linear(dim_model * 3, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * ffn_multiplier, dim_model),
        )

    def _causal_depthwise_conv(self, conv: nn.Conv1d, inputs: Tensor) -> Tensor:
        channels_first = inputs.transpose(1, 2)
        padded = F.pad(channels_first, (conv.kernel_size[0] - 1, 0))
        return conv(padded).transpose(1, 2)

    def _causal_linear_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        mask_4d = mask.unsqueeze(-1).unsqueeze(-1).to(v.dtype)
        q = (F.elu(q) + 1.0) * mask_4d
        k = (F.elu(k) + 1.0) * mask_4d
        v = v * mask_4d

        kv = torch.einsum("bthd,bthe->bthde", k, v)
        kv_prefix = kv.cumsum(dim=1)
        k_prefix = k.cumsum(dim=1)
        numerators = torch.einsum("bthd,bthde->bthe", q, kv_prefix)
        denominators = (q * k_prefix).sum(dim=-1, keepdim=True)
        outputs = numerators / (denominators + 1e-6)
        return outputs * mask_4d

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        residual = inputs
        normed = self.pre_norm(inputs)
        gate_values, q, k, v = torch.split(
            self.input_proj(normed),
            [self.dim_model, self.inner_dim, self.inner_dim, self.inner_dim],
            dim=-1,
        )
        batch_size, seq_len, _ = inputs.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        content = self._causal_linear_attention(q, k, v, mask).reshape(batch_size, seq_len, self.inner_dim)
        content = self.content_proj(content)
        temporal = self._causal_depthwise_conv(self.temporal_conv, normed)
        positional = self._causal_depthwise_conv(self.positional_conv, normed)

        mixed = torch.cat((content, temporal, positional), dim=-1)
        mixed = self.mix_proj(mixed)
        mask_f = mask.unsqueeze(-1).to(inputs.dtype)
        outputs = residual + self.dropout(torch.sigmoid(gate_values) * mixed)
        outputs = outputs * mask_f
        outputs = outputs + self.dropout(self.ffn(self.ffn_norm(outputs)))
        return outputs * mask_f


class FuXiLinearSequenceModel(BaseSequenceRecommender):
    def __init__(self, num_items: int, max_history: int) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_history = max_history
        self.dim_model = 256
        self.num_gap_buckets = 64
        self.num_recency_buckets = 64

        self.feature_builder = FuXiSequenceFeatures(
            max_history=max_history,
            num_gap_buckets=self.num_gap_buckets,
            num_recency_buckets=self.num_recency_buckets,
        )

        self.item_embedding = nn.Embedding(num_items + 1, self.dim_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history, self.dim_model)
        self.gap_embedding = nn.Embedding(self.num_gap_buckets, self.dim_model)
        self.recency_embedding = nn.Embedding(self.num_recency_buckets, self.dim_model)
        self.time_feature_proj = nn.Sequential(
            nn.Linear(2, self.dim_model),
            nn.GELU(),
            nn.Linear(self.dim_model, self.dim_model),
        )
        self.input_norm = nn.LayerNorm(self.dim_model)
        self.input_dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [
                FuXiLinearBlock(
                    dim_model=self.dim_model,
                    num_heads=4,
                    head_dim=32,
                    dropout=0.1,
                    ffn_multiplier=4,
                    temporal_kernel_size=3,
                    positional_kernel_size=5,
                )
                for _ in range(8)
            ]
        )
        self.final_norm = nn.LayerNorm(self.dim_model)
        self.pool_gate = nn.Linear(self.dim_model, 1)
        self.user_head = nn.Sequential(
            nn.Linear(self.dim_model * 3, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, self.dim_model),
        )
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))

    def _build_token_embeddings(
        self,
        history_ids: Tensor,
        history_timestamps: Tensor,
        history_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gap_buckets, recency_buckets, mask = self.feature_builder.build(
            history_timestamps=history_timestamps,
            history_lengths=history_lengths,
        )
        positions = torch.arange(self.max_history, device=history_ids.device)
        position_embeddings = self.position_embedding(positions).unsqueeze(0)
        item_embeddings = self.item_embedding(history_ids)
        gap_embeddings = self.gap_embedding(gap_buckets)
        recency_embeddings = self.recency_embedding(recency_buckets)

        latest_timestamps = history_timestamps[:, -1:]
        gap_values = gap_buckets.to(torch.float32)
        recency_values = (latest_timestamps - history_timestamps).clamp_min(0).to(torch.float32)
        time_features = torch.stack(
            (
                torch.log1p(gap_values),
                torch.log1p(recency_values),
            ),
            dim=-1,
        )
        dense_time = self.time_feature_proj(time_features)

        token_embeddings = item_embeddings + position_embeddings + gap_embeddings + recency_embeddings + dense_time
        token_embeddings = self.input_dropout(self.input_norm(token_embeddings))
        return token_embeddings * mask.unsqueeze(-1).to(token_embeddings.dtype), mask

    def encode_users(
        self,
        history_ids: Tensor,
        history_timestamps: Tensor,
        history_lengths: Tensor,
    ) -> Tensor:
        outputs, mask = self._build_token_embeddings(
            history_ids=history_ids,
            history_timestamps=history_timestamps,
            history_lengths=history_lengths,
        )
        for layer in self.layers:
            outputs = layer(outputs, mask)
        outputs = self.final_norm(outputs) * mask.unsqueeze(-1).to(outputs.dtype)

        mask_f = mask.unsqueeze(-1).to(outputs.dtype)
        denom = history_lengths.clamp_min(1).unsqueeze(-1).to(outputs.dtype)
        mean_pooled = (outputs * mask_f).sum(dim=1) / denom

        pool_scores = self.pool_gate(outputs).squeeze(-1)
        pool_scores = pool_scores.masked_fill(~mask, -1e9)
        pool_weights = torch.softmax(pool_scores, dim=1).unsqueeze(-1)
        attn_pooled = (pool_weights * outputs).sum(dim=1)

        last_hidden = outputs[:, -1, :]
        user_embeddings = self.user_head(torch.cat((last_hidden, mean_pooled, attn_pooled), dim=-1))
        return F.normalize(user_embeddings, dim=-1)

    def score_all_items(self, user_embeddings: Tensor) -> Tensor:
        item_embeddings = self.item_embedding.weight[1 : self.num_items + 1]
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        return user_embeddings @ item_embeddings.t() + self.item_bias[1 : self.num_items + 1]

    def score_candidates(self, user_embeddings: Tensor, candidate_ids: Tensor) -> Tensor:
        item_embeddings = self.item_embedding(candidate_ids)
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        scores = (user_embeddings.unsqueeze(1) * item_embeddings).sum(dim=-1)
        return scores + self.item_bias[candidate_ids]


def build_model(num_items: int, max_history: int) -> BaseSequenceRecommender:
    return FuXiLinearSequenceModel(num_items=num_items, max_history=max_history)


# RegexTagCustomPruningAlgorithmEnd


def run_syntax_only() -> None:
    model = build_model(num_items=256, max_history=MAX_HISTORY)
    ensure_model_contract(model)

    history_ids = torch.randint(1, 128, (8, MAX_HISTORY))
    history_ids[:, : MAX_HISTORY // 2] = 0
    history_timestamps = torch.arange(MAX_HISTORY).repeat(8, 1)
    history_lengths = torch.full((8,), MAX_HISTORY // 2, dtype=torch.long)
    user_embeddings = model.encode_users(history_ids, history_timestamps, history_lengths)
    candidate_ids = torch.randint(1, 128, (8, 16))
    logits = score_training_candidates(model, user_embeddings, candidate_ids)
    assert user_embeddings.ndim == 2
    assert logits.shape == (8, 16)
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
        "hr@10": 0.0,
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
    hr10_sum = 0.0
    mrr_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            history_ids = batch["history_ids"].to(device)
            history_timestamps = batch["history_timestamps"].to(device)
            history_lengths = batch["history_lengths"].to(device)
            target_ids = batch["target_ids"].to(device)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                user_embeddings = model.encode_users(history_ids, history_timestamps, history_lengths)
                scores = model.score_all_items(user_embeddings).float()
            if not all_finite(user_embeddings):
                return invalid_metrics("Non-finite user embeddings encountered during evaluation.")
            if not all_finite(scores):
                return invalid_metrics("Non-finite item scores encountered during evaluation.")

            target_index = target_ids - 1
            for row_idx in range(scores.size(0)):
                seen = history_ids[row_idx]
                seen = seen[seen > 0] - 1
                seen = seen[seen != target_index[row_idx]]
                if seen.numel() > 0:
                    scores[row_idx, seen] = -1e9

            target_scores = scores.gather(1, target_index.unsqueeze(1)).squeeze(1)
            if not all_finite(target_scores):
                return invalid_metrics("Non-finite target scores encountered during evaluation.")
            ranks = 1 + (scores > target_scores.unsqueeze(1)).sum(dim=1)
            mrr_sum += (1.0 / ranks.float()).sum().item()

            topk_indices = scores.topk(10, dim=1).indices
            hits = topk_indices.eq(target_index.unsqueeze(1))
            hr10_sum += hits.any(dim=1).float().sum().item()

            hit_rows = hits.any(dim=1)
            if hit_rows.any():
                hit_positions = hits[hit_rows].float().argmax(dim=1).float()
                ndcg10_sum += (1.0 / torch.log2(hit_positions + 2.0)).sum().item()

            count += int(target_ids.size(0))

    ndcg10 = ndcg10_sum / max(count, 1)
    hr10 = hr10_sum / max(count, 1)
    mrr = mrr_sum / max(count, 1)
    combined = (ndcg10 + hr10 + mrr) / 3.0
    return {
        "combined_score": combined,
        "ndcg@10": ndcg10,
        "hr@10": hr10,
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
            negative_ids = sample_negatives(target_ids, prepared.num_items, NUM_NEGATIVES)
            candidate_ids = torch.cat((target_ids.unsqueeze(1), negative_ids), dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                user_embeddings = model.encode_users(history_ids, history_timestamps, history_lengths)
                logits = score_training_candidates(model, user_embeddings, candidate_ids)
                loss = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=device))
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
