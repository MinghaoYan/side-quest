"""Fixed-budget KuaRec training script with an editable Wukong-style model block."""

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
    cache_path = f"{dataset_csv}.kuairec_wukong_cache.pt"
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

class SequenceFeatureBuilder:
    def __init__(
        self,
        num_items: int,
        max_history: int,
        recent_slots: int = 20,
        frequent_slots: int = 8,
        gap_slots: int = 6,
        num_gap_buckets: int = 16,
        num_stat_buckets: int = 12,
    ) -> None:
        self.num_items = num_items
        self.max_history = max_history
        self.recent_slots = recent_slots
        self.frequent_slots = frequent_slots
        self.gap_slots = gap_slots
        self.num_gap_buckets = num_gap_buckets
        self.num_stat_buckets = num_stat_buckets

        self.gap_offset = num_items + 1
        self.length_offset = self.gap_offset + num_gap_buckets
        self.ratio_offset = self.length_offset + num_stat_buckets

        self.total_vocab_size = self.ratio_offset + num_stat_buckets
        self.num_sparse_fields = recent_slots + frequent_slots + gap_slots + 2
        self.dense_dim = 10

    def _bucketize_gap(self, value: float) -> int:
        if value <= 0:
            return self.gap_offset
        bucket = int(min(self.num_gap_buckets - 1, math.log2(value + 1.0)))
        return self.gap_offset + bucket

    def _bucketize_stat(self, value: float) -> int:
        clipped = min(self.num_stat_buckets - 1, max(0, int(value * self.num_stat_buckets)))
        return clipped

    def build(
        self,
        history_ids: Tensor,
        history_timestamps: Tensor,
        history_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        device = history_ids.device
        batch_size = int(history_ids.size(0))

        sparse = torch.zeros(
            batch_size,
            self.num_sparse_fields,
            dtype=torch.long,
            device=device,
        )
        dense = torch.zeros(batch_size, self.dense_dim, dtype=torch.float32, device=device)

        recent_items = torch.flip(history_ids[:, -self.recent_slots :], dims=[1])
        sparse[:, : self.recent_slots] = recent_items

        for row_idx in range(batch_size):
            length = int(history_lengths[row_idx].item())
            if length <= 0:
                continue

            items = history_ids[row_idx, -length:]
            times = history_timestamps[row_idx, -length:]

            unique_items, counts = items.unique(return_counts=True)
            order = torch.argsort(counts, descending=True)
            top_items = unique_items[order][: self.frequent_slots]
            start = self.recent_slots
            sparse[row_idx, start : start + top_items.numel()] = top_items

            gaps = torch.diff(times.to(torch.float32)) if length > 1 else torch.zeros(0, device=device)
            recent_gaps = gaps[-self.gap_slots :]
            gap_start = self.recent_slots + self.frequent_slots
            for gap_idx, gap_value in enumerate(recent_gaps):
                sparse[row_idx, gap_start + gap_idx] = self._bucketize_gap(float(gap_value.item()))

            unique_ratio = float(unique_items.numel()) / float(length)
            repeat_ratio = 1.0 - unique_ratio
            length_bucket = min(self.num_stat_buckets - 1, int(math.log2(length + 1)))
            ratio_bucket = self._bucketize_stat(unique_ratio)
            sparse[row_idx, -2] = self.length_offset + length_bucket
            sparse[row_idx, -1] = self.ratio_offset + ratio_bucket

            mean_gap = float(gaps.mean().item()) if gaps.numel() > 0 else 0.0
            std_gap = float(gaps.std(unbiased=False).item()) if gaps.numel() > 1 else 0.0
            last_gap = float(gaps[-1].item()) if gaps.numel() > 0 else 0.0

            dense[row_idx, 0] = float(length) / float(self.max_history)
            dense[row_idx, 1] = math.log1p(float(length))
            dense[row_idx, 2] = unique_ratio
            dense[row_idx, 3] = repeat_ratio
            dense[row_idx, 4] = math.log1p(mean_gap)
            dense[row_idx, 5] = math.log1p(std_gap)
            dense[row_idx, 6] = math.log1p(last_gap)
            dense[row_idx, 7] = float(items[-1].item()) / float(self.num_items)
            dense[row_idx, 8] = math.log1p(float(times[-1].item() - times[0].item()))
            dense[row_idx, 9] = float(counts.max().item()) / float(length)

        return sparse, dense


class LinearCompressBlock(nn.Module):
    def __init__(self, num_emb_in: int, num_emb_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_emb_in, num_emb_out)))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs.permute(0, 2, 1)
        outputs = outputs @ self.weight
        return outputs.permute(0, 2, 1)


class FactorizationMachineBlock(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        self.weight = nn.Parameter(torch.empty((num_emb_in, rank)))
        self.norm = nn.LayerNorm(num_emb_in * rank)
        self.mlp = nn.Sequential(
            nn.Linear(num_emb_in * rank, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emb_out * dim_emb),
        )
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs.permute(0, 2, 1)
        outputs = outputs @ self.weight
        outputs = torch.bmm(inputs, outputs)
        outputs = outputs.reshape(-1, self.num_emb_in * self.rank)
        outputs = self.mlp(self.norm(outputs))
        return outputs.reshape(-1, self.num_emb_out, self.dim_emb)


class ResidualProjection(nn.Module):
    def __init__(self, num_emb_in: int, num_emb_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_emb_in, num_emb_out)))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs.permute(0, 2, 1)
        outputs = outputs @ self.weight
        return outputs.permute(0, 2, 1)


class WukongLayer(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lcb = LinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = FactorizationMachineBlock(
            num_emb_in=num_emb_in,
            num_emb_out=num_emb_fmb,
            dim_emb=dim_emb,
            rank=rank_fmb,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim_emb)
        out_fields = num_emb_lcb + num_emb_fmb
        if out_fields == num_emb_in:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = ResidualProjection(num_emb_in, out_fields)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = torch.cat((self.fmb(inputs), self.lcb(inputs)), dim=1)
        return self.norm(outputs + self.residual_projection(inputs))


class WukongSequenceModel(BaseSequenceRecommender):
    def __init__(self, num_items: int, max_history: int) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_history = max_history
        self.dim_emb = 288

        self.feature_builder = SequenceFeatureBuilder(num_items=num_items, max_history=max_history)
        self.sparse_embedding = nn.Embedding(
            self.feature_builder.total_vocab_size + 1,
            self.dim_emb,
            padding_idx=0,
        )
        self.dense_embedding = nn.Linear(
            self.feature_builder.dense_dim,
            self.feature_builder.dense_dim * self.dim_emb,
        )

        num_fields = self.feature_builder.num_sparse_fields + self.feature_builder.dense_dim
        self.layers = nn.ModuleList(
            [
                WukongLayer(
                    num_emb_in=num_fields,
                    dim_emb=self.dim_emb,
                    num_emb_lcb=32,
                    num_emb_fmb=32,
                    rank_fmb=16,
                    hidden_dim=4608,
                    dropout=0.15,
                ),
                WukongLayer(
                    num_emb_in=64,
                    dim_emb=self.dim_emb,
                    num_emb_lcb=32,
                    num_emb_fmb=32,
                    rank_fmb=16,
                    hidden_dim=4608,
                    dropout=0.15,
                ),
                WukongLayer(
                    num_emb_in=64,
                    dim_emb=self.dim_emb,
                    num_emb_lcb=32,
                    num_emb_fmb=32,
                    rank_fmb=16,
                    hidden_dim=4608,
                    dropout=0.15,
                ),
                WukongLayer(
                    num_emb_in=64,
                    dim_emb=self.dim_emb,
                    num_emb_lcb=32,
                    num_emb_fmb=32,
                    rank_fmb=16,
                    hidden_dim=4608,
                    dropout=0.15,
                ),
                WukongLayer(
                    num_emb_in=64,
                    dim_emb=self.dim_emb,
                    num_emb_lcb=32,
                    num_emb_fmb=32,
                    rank_fmb=16,
                    hidden_dim=4608,
                    dropout=0.15,
                ),
                WukongLayer(
                    num_emb_in=64,
                    dim_emb=self.dim_emb,
                    num_emb_lcb=32,
                    num_emb_fmb=32,
                    rank_fmb=16,
                    hidden_dim=4608,
                    dropout=0.15,
                ),
            ]
        )
        self.field_gate = nn.Linear(self.dim_emb, 1)
        self.user_head = nn.Sequential(
            nn.Linear(64 * self.dim_emb + 2 * self.dim_emb, 2304),
            nn.GELU(),
            nn.LayerNorm(2304),
            nn.Linear(2304, 1152),
            nn.GELU(),
            nn.LayerNorm(1152),
            nn.Linear(1152, self.dim_emb),
        )
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))

    def encode_users(
        self,
        history_ids: Tensor,
        history_timestamps: Tensor,
        history_lengths: Tensor,
    ) -> Tensor:
        sparse_inputs, dense_inputs = self.feature_builder.build(
            history_ids=history_ids,
            history_timestamps=history_timestamps,
            history_lengths=history_lengths,
        )
        sparse_outputs = self.sparse_embedding(sparse_inputs)
        dense_outputs = self.dense_embedding(dense_inputs).view(
            -1, self.feature_builder.dense_dim, self.dim_emb
        )
        outputs = torch.cat((sparse_outputs, dense_outputs), dim=1)
        for layer in self.layers:
            outputs = layer(outputs)

        field_weights = torch.softmax(self.field_gate(outputs).squeeze(-1), dim=1)
        pooled = (field_weights.unsqueeze(-1) * outputs).sum(dim=1)
        mean_pooled = outputs.mean(dim=1)
        flat = outputs.flatten(1)
        user_embeddings = self.user_head(torch.cat((flat, pooled, mean_pooled), dim=1))
        return F.normalize(user_embeddings, dim=-1)

    def score_all_items(self, user_embeddings: Tensor) -> Tensor:
        item_embeddings = self.sparse_embedding.weight[1 : self.num_items + 1]
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        return user_embeddings @ item_embeddings.t() + self.item_bias[1 : self.num_items + 1]

    def score_candidates(self, user_embeddings: Tensor, candidate_ids: Tensor) -> Tensor:
        item_embeddings = self.sparse_embedding(candidate_ids)
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        scores = (user_embeddings.unsqueeze(1) * item_embeddings).sum(dim=-1)
        return scores + self.item_bias[candidate_ids]


def build_model(num_items: int, max_history: int) -> BaseSequenceRecommender:
    return WukongSequenceModel(num_items=num_items, max_history=max_history)


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
    parser = argparse.ArgumentParser(description="Train and evaluate the fixed-budget KuaRec Wukong baseline.")
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
