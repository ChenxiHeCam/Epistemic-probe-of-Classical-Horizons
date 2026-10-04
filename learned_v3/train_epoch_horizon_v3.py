"""Train the classical-only EPOCH dual encoder on a frozen horizon view."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from epoch_formula_graph_dataset import (  # noqa: E402
    EpochFormulaGraphStore,
    PROVENANCE_MODES,
)
from epoch_learned_v3 import (  # noqa: E402
    EpochArchitecture,
    EpochDualEncoder,
    cross_modal_loss,
    parameter_count,
    view_regularization,
)
from epoch_pointcloud_dataset import make_loader  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def contrastive_identity_lookup(
    graph_store: EpochFormulaGraphStore,
    pointcloud_bundle: str | Path,
    allowed_horizons: list[str] | None = None,
) -> tuple[torch.Tensor, list[str], Path]:
    """Build physics-aware positives without conflating broad leakage components.

    Source-audited/registry-linked pre-1900 formulas use their historical law
    family.  The generator-gated 1901--1950 corpus has no dated family registry,
    so its much narrower temporal relation component is used.  Unresolved
    pre-1900 formulas fall back to exact formula identity for expanded-only
    sensitivity runs.  Leakage groups still govern train/test splitting; they
    are deliberately not treated as semantic labels because some are broad
    connected components spanning several physical families.
    """
    formula_table = Path(pointcloud_bundle).resolve() / "formulas.jsonl.gz"
    allowed = set(allowed_horizons) if allowed_horizons else None
    identities: list[str | None] = []
    with gzip.open(formula_table, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != len(graph_store.metadata):
        raise RuntimeError("point-cloud/formula-graph table cardinality mismatch")
    for index, (row, metadata) in enumerate(zip(rows, graph_store.metadata)):
        if (
            int(metadata["formula_index"]) != index
            or row["pointcloud_formula_id"] != metadata["pointcloud_formula_id"]
        ):
            raise RuntimeError(f"point-cloud/formula-graph identity mismatch at {index}")
        if allowed is not None and row["horizon"] not in allowed:
            identities.append(None)
            continue
        historical = sorted(set(map(str, row.get("historical_families") or [])))
        if historical:
            # A tiny number of exact expressions are linked to multiple laws;
            # retain the full set instead of choosing a label post hoc.
            identity = "historical_family_set:" + "|".join(historical)
        elif row["horizon"] == "pre1950":
            identity = "temporal_relation_group:" + str(row["leakage_group_id"])
        else:
            identity = "unresolved_exact_formula:" + str(row["pointcloud_formula_id"])
        identities.append(identity)
    names = sorted({name for name in identities if name is not None})
    mapping = {name: index for index, name in enumerate(names)}
    return torch.tensor([
        mapping[name] if name is not None else -1 for name in identities
    ]), names, formula_table


class EmbeddingQueue:
    def __init__(self, capacity: int, dimension: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.embeddings = torch.zeros(capacity, dimension, device=device)
        self.formula_ids = torch.full(
            (capacity,), -1, dtype=torch.long, device=device
        )
        self.pointer = 0
        self.size = 0

    def values(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.size == 0:
            return None, None
        return self.embeddings[:self.size], self.formula_ids[:self.size]

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, formula_ids: torch.Tensor) -> None:
        embeddings = F.normalize(embeddings.detach(), dim=-1)
        formula_ids = formula_ids.detach()
        if len(embeddings) >= self.capacity:
            self.embeddings.copy_(embeddings[-self.capacity:])
            self.formula_ids.copy_(formula_ids[-self.capacity:])
            self.pointer = 0
            self.size = self.capacity
            return
        count = len(embeddings)
        first = min(count, self.capacity - self.pointer)
        self.embeddings[self.pointer:self.pointer + first] = embeddings[:first]
        self.formula_ids[self.pointer:self.pointer + first] = formula_ids[:first]
        remaining = count - first
        if remaining:
            self.embeddings[:remaining] = embeddings[first:]
            self.formula_ids[:remaining] = formula_ids[first:]
        self.pointer = (self.pointer + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "embeddings": self.embeddings[:self.size].detach().cpu(),
            "formula_ids": self.formula_ids[:self.size].detach().cpu(),
            "pointer": self.pointer,
            "size": self.size,
        }

    def load_state_dict(self, state: dict) -> None:
        size = min(int(state["size"]), self.capacity)
        self.embeddings[:size].copy_(state["embeddings"][:size].to(self.embeddings))
        self.formula_ids[:size].copy_(state["formula_ids"][:size].to(self.formula_ids))
        self.size = size
        self.pointer = int(state["pointer"]) % self.capacity


def point_views(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, count, _ = points.shape
    views = []
    masks = []
    for _ in range(2):
        keep_probability = torch.empty(batch, 1, device=points.device).uniform_(0.82, 1.0)
        mask = torch.rand(batch, count, device=points.device) < keep_probability
        mask[:, 0] = True
        scale = points.std(dim=1, keepdim=True).clamp_min(1e-3)
        noise_level = torch.empty(batch, 1, 1, device=points.device).uniform_(0.0, 0.012)
        view = points + torch.randn_like(points) * scale * noise_level
        views.append(view)
        masks.append(mask)
    return views[0], masks[0], views[1], masks[1]


def to_device_graph(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validation_alignment(
    model: EpochDualEncoder,
    loader,
    graph_store: EpochFormulaGraphStore,
    formula_to_group: torch.Tensor,
    device: torch.device,
    precision: torch.dtype,
    max_batches: int,
) -> dict:
    model.eval()
    losses = []
    similarities = []
    records = 0
    for batch_index, point_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        points = point_batch["points"].to(device, non_blocking=True)
        formula_ids = point_batch["formula_index"].to(device, non_blocking=True)
        point_group_ids = formula_to_group[formula_ids]
        graph = to_device_graph(
            graph_store.batch(point_batch["formula_index"], deduplicate=True), device
        )
        real_mask = torch.ones(points.shape[:2], dtype=torch.bool, device=device)
        with torch.autocast("cuda", dtype=precision):
            point_embedding = model.point_encoder(points, real_mask)
        with torch.autocast("cuda", enabled=False):
            graph_embedding = model.formula_encoder(graph)
        with torch.autocast("cuda", dtype=precision):
            loss, _ = cross_modal_loss(
                point_embedding, point_group_ids, graph_embedding,
                formula_to_group[graph["graph_formula_index"]], model.temperature,
            )
            matched = graph_embedding[graph["sample_to_graph"]]
            similarity = F.cosine_similarity(point_embedding, matched, dim=-1).mean()
        losses.append(float(loss))
        similarities.append(float(similarity))
        records += len(points)
    model.train()
    return {
        "records": records,
        "batches": len(losses),
        "alignment_loss": float(np.mean(losses)) if losses else None,
        "matched_cosine": float(np.mean(similarities)) if similarities else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--queue-size", type=int, default=16384)
    parser.add_argument("--view-weight", type=float, default=0.25)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--benchmark-only", action="store_true",
        help="record throughput/metrics without writing model or optimizer checkpoints",
    )
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--phase-a-freeze", type=Path)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--point-layers", type=int, default=6)
    parser.add_argument("--graph-local-layers", type=int, default=4)
    parser.add_argument("--graph-global-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--kan-layer", type=int, default=1)
    parser.add_argument("--kan-bottleneck", type=int, default=96)
    parser.add_argument("--kan-grid", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal training run")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["knowledge_cutoff"] == 1950:
        if args.phase_a_freeze is None or not args.phase_a_freeze.resolve().is_file():
            raise RuntimeError(
                "pre1950 joint training is fail-closed until --phase-a-freeze points "
                "to a frozen phase-A result"
            )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    precision = torch.bfloat16

    point_config = config["pointcloud"]
    horizons = ["pre1900"] if config["knowledge_cutoff"] == 1899 else None
    dataset, sampler, loader = make_loader(
        point_config["bundle"], point_config["view_index"], point_config["view"],
        "train", point_config["headline_provenance_mode"], args.batch_size,
        args.points, args.workers, shuffle=True, seed=args.seed, horizons=horizons,
    )
    _cal_dataset, _cal_sampler, cal_loader = make_loader(
        point_config["bundle"], point_config["view_index"], point_config["view"],
        "calibration", point_config["headline_provenance_mode"], args.batch_size,
        args.points, min(args.workers, 4), shuffle=False, seed=args.seed,
        horizons=horizons,
    )
    graph_store = EpochFormulaGraphStore(config["formula_graph"]["bundle"])
    formula_to_group, group_names, formula_table = contrastive_identity_lookup(
        graph_store, point_config["bundle"], horizons
    )
    role_key = f"{point_config['view']}_role"
    training_formula_indices = torch.tensor([
        int(row["formula_index"])
        for row in graph_store.metadata
        if row[role_key] == "train"
        and row["tier"] in PROVENANCE_MODES[
            point_config["headline_provenance_mode"]
        ]
        and (horizons is None or row["horizon"] in horizons)
    ], dtype=torch.long)
    training_identity_count = int(
        formula_to_group[training_formula_indices].unique().numel()
    )
    if bool((formula_to_group[training_formula_indices] < 0).any()):
        raise RuntimeError("training formula lacks an allowed-horizon identity")
    formula_to_group = formula_to_group.to(device=device, non_blocking=True)
    architecture = EpochArchitecture(
        d_model=args.d_model,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        point_layers=args.point_layers,
        graph_local_layers=args.graph_local_layers,
        graph_global_layers=args.graph_global_layers,
        dropout=args.dropout,
        kan_layer=args.kan_layer,
        kan_bottleneck=args.kan_bottleneck,
        kan_grid=args.kan_grid,
    )
    model = EpochDualEncoder(
        graph_store.manifest["node_vocab_size"], architecture
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), fused=True,
    )
    batches_per_epoch = len(sampler)
    requested_batches = args.epochs * batches_per_epoch
    if args.max_steps:
        requested_batches = min(requested_batches, args.max_steps)
    total_optimizer_steps = max(1, math.ceil(requested_batches / args.grad_accum))

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max(1e-3, (step + 1) / max(1, args.warmup_steps))
        progress = min(
            1.0,
            (step - args.warmup_steps)
            / max(1, total_optimizer_steps - args.warmup_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    queue = EmbeddingQueue(
        args.queue_size, architecture.embedding_dim, device
    )
    global_batch = 0
    optimizer_step = 0
    start_epoch = 0
    resume_batch_in_epoch = 0
    if args.resume:
        checkpoint = torch.load(
            args.resume.resolve(), map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        queue.load_state_dict(checkpoint["queue"])
        global_batch = int(checkpoint["global_batch"])
        optimizer_step = int(checkpoint["optimizer_step"])
        start_epoch = int(checkpoint["epoch"])
        resume_batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))

    provenance = {
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "pointcloud_manifest_sha256": sha256(
            Path(point_config["bundle"]) / "manifest.json"
        ),
        "view_manifest_sha256": sha256(
            Path(point_config["view_index"]) / "manifest.json"
        ),
        "graph_manifest_sha256": sha256(
            Path(config["formula_graph"]["bundle"]) / "manifest.json"
        ),
        "pointcloud_formula_table_sha256": sha256(formula_table),
        "phase_a_freeze_sha256": (
            sha256(args.phase_a_freeze.resolve()) if args.phase_a_freeze else None
        ),
    }
    run_spec = {
        "run_id": config["run_id"],
        "knowledge_cutoff": config["knowledge_cutoff"],
        "classical_only": True,
        "breakdown_or_deformation_supervision": False,
        "architecture": architecture.to_dict(),
        "model_parameters": parameter_count(model),
        "contrastive_identity": (
            "historical family-set when available; temporal relation group for "
            "pre1950 candidates; exact formula for unresolved pre1900"
        ),
        "allowed_horizon_contrastive_identities": len(group_names),
        "training_contrastive_identities": training_identity_count,
        "dataset_records": len(dataset),
        "batches_per_epoch": batches_per_epoch,
        "optimization": {
            "epochs": args.epochs, "max_steps": args.max_steps,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps, "queue_size": args.queue_size,
            "precision": "bfloat16", "seed": args.seed,
            "benchmark_only": args.benchmark_only,
        },
        "provenance": provenance,
    }
    (output / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_spec, indent=2), flush=True)

    stop = {"requested": False}

    def request_stop(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    metrics_path = output / "metrics.jsonl"

    def checkpoint_payload(epoch: int, batch_in_epoch: int) -> dict:
        return {
            "schema_version": "3.0",
            "run_spec": run_spec,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "queue": queue.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_batch": global_batch,
            "optimizer_step": optimizer_step,
        }

    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()
    interval_time = start_time
    interval_records = 0
    last_metrics = None
    completed_epoch = start_epoch
    checkpoint_epoch = start_epoch
    checkpoint_batch_in_epoch = resume_batch_in_epoch
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch_index, point_batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch_in_epoch:
                continue
            if args.max_steps and global_batch >= args.max_steps:
                stop["requested"] = True
                break
            points = point_batch["points"].to(device, non_blocking=True)
            formula_ids = point_batch["formula_index"].to(device, non_blocking=True)
            point_group_ids = formula_to_group[formula_ids]
            graph = to_device_graph(
                graph_store.batch(point_batch["formula_index"], deduplicate=True), device
            )
            first, first_mask, second, second_mask = point_views(points)
            with torch.autocast("cuda", dtype=precision):
                first_embedding = model.point_encoder(first, first_mask)
                second_embedding = model.point_encoder(second, second_mask)
            # Graphs contain at most 71 nodes and are inexpensive in FP32.
            # Keep this branch full precision for stable KAN bases and to avoid
            # a torch 2.1/cu121 H20 BF16 Linear-kernel SIGFPE.
            with torch.autocast("cuda", enabled=False):
                graph_embedding = model.formula_encoder(graph)
            graph_group_ids = formula_to_group[graph["graph_formula_index"]]
            with torch.autocast("cuda", dtype=precision):
                all_points = torch.cat((first_embedding, second_embedding), dim=0)
                all_group_ids = torch.cat((point_group_ids, point_group_ids), dim=0)
                queue_embeddings, queue_ids = queue.values()
                alignment, alignment_parts = cross_modal_loss(
                    all_points, all_group_ids, graph_embedding,
                    graph_group_ids, model.temperature,
                    queue_embeddings, queue_ids,
                )
                view_loss, view_parts = view_regularization(
                    first_embedding, second_embedding
                )
                loss = alignment + args.view_weight * view_loss
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            queue.enqueue(graph_embedding, graph_group_ids)
            global_batch += 1
            checkpoint_epoch = epoch
            checkpoint_batch_in_epoch = batch_index + 1
            interval_records += len(points)
            grad_norm = None
            if global_batch % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                with torch.no_grad():
                    model.logit_scale.clamp_(0.0, math.log(100.0))

            matched_graph = graph_embedding[graph["sample_to_graph"]]
            matched_cosine = 0.5 * (
                F.cosine_similarity(first_embedding.detach(), matched_graph.detach()).mean()
                + F.cosine_similarity(second_embedding.detach(), matched_graph.detach()).mean()
            )
            last_metrics = {
                "event": "train",
                "epoch": epoch,
                "global_batch": global_batch,
                "optimizer_step": optimizer_step,
                "loss": float(loss.detach()),
                "alignment": float(alignment.detach()),
                "point_to_graph": float(alignment_parts["point_to_graph"]),
                "graph_to_point": float(alignment_parts["graph_to_point"]),
                "view_cosine_loss": float(view_parts["view_cosine"]),
                "variance_penalty": float(view_parts["variance"]),
                "matched_cosine": float(matched_cosine),
                "temperature_scale": float(model.temperature.detach()),
                "unique_graphs": int(graph["graph_formula_index"].numel()),
                "unique_groups": int(graph_group_ids.unique().numel()),
                "queue_size": queue.size,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
            }
            if global_batch == 1 or global_batch % args.log_every == 0:
                now = time.perf_counter()
                last_metrics["records_per_second"] = interval_records / max(
                    1e-9, now - interval_time
                )
                last_metrics["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(last_metrics) + "\n")
                print(json.dumps(last_metrics), flush=True)
                interval_time = now
                interval_records = 0
            if (
                not args.benchmark_only
                and args.save_every
                and global_batch % args.save_every == 0
            ):
                atomic_torch_save(
                    checkpoint_payload(
                        checkpoint_epoch, checkpoint_batch_in_epoch
                    ), output / f"checkpoint_{global_batch:07d}.pt"
                )
                atomic_torch_save(
                    checkpoint_payload(
                        checkpoint_epoch, checkpoint_batch_in_epoch
                    ), output / "checkpoint_last.pt"
                )
            if stop["requested"]:
                break
        if checkpoint_batch_in_epoch >= batches_per_epoch:
            completed_epoch = epoch + 1
            checkpoint_epoch = epoch + 1
            checkpoint_batch_in_epoch = 0
            resume_batch_in_epoch = 0
        else:
            completed_epoch = epoch
        validation = validation_alignment(
            model, cal_loader, graph_store, formula_to_group, device, precision,
            args.validation_batches,
        )
        validation.update({
            "event": "validation", "epoch": epoch,
            "global_batch": global_batch, "optimizer_step": optimizer_step,
        })
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validation) + "\n")
        print(json.dumps(validation), flush=True)
        if not args.benchmark_only:
            atomic_torch_save(
                checkpoint_payload(
                    checkpoint_epoch, checkpoint_batch_in_epoch
                ), output / "checkpoint_last.pt"
            )
        if stop["requested"]:
            break

    elapsed = time.perf_counter() - start_time
    final_checkpoint = output / "checkpoint_final.pt"
    if not args.benchmark_only:
        atomic_torch_save(
            checkpoint_payload(
                checkpoint_epoch, checkpoint_batch_in_epoch
            ), final_checkpoint
        )
    summary = {
        "status": "completed" if not stop["requested"] or (
            args.max_steps and global_batch >= args.max_steps
        ) else "stopped",
        "run_id": config["run_id"],
        "knowledge_cutoff": config["knowledge_cutoff"],
        "global_batches": global_batch,
        "optimizer_steps": optimizer_step,
        "epochs_completed": completed_epoch,
        "resume_epoch": checkpoint_epoch,
        "resume_batch_in_epoch": checkpoint_batch_in_epoch,
        "elapsed_seconds": elapsed,
        "mean_clouds_per_second": global_batch * args.batch_size / max(elapsed, 1e-9),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "final_checkpoint": str(final_checkpoint) if not args.benchmark_only else None,
        "final_checkpoint_sha256": (
            sha256(final_checkpoint) if not args.benchmark_only else None
        ),
        "last_metrics": last_metrics,
        "future_horizon_used_for_optimization": False,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
