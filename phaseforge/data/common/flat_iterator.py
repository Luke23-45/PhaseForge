"""Flat in-memory batch iterator for the single-step (non-RNN) protocol.

Training review T5 (``docs/dev/training_performance_review.md`` §2.5). For
``sequence_length == 1 and stride == 1`` — every non-RNN cell of the
protocol — the whole data split is a fixed set of rows. The per-item
``__getitem__`` + collate path costs ~225 µs/sample of pure Python/torch-op
dispatch (measured), and hiding it behind DataLoader workers leaves a
~12 ms/batch worker-IPC floor plus a per-process worker spawn. This iterator
concatenates the split ONCE into six flat tensors and produces each batch
with one ``randperm`` slice + one gather per field: zero per-item Python, no
workers, no IPC, no spawn.

Content contract
----------------
* Batch keys, shapes, dtypes and VALUES are identical to
  :class:`phaseforge.data.common.collator.PhaseAwareCollator` on the same
  ``StateOnlyDataset`` (built after phase corruption, so corrupted labels
  flow through unchanged).
* ``drop_last`` parity with ``DataLoader(drop_last=...)``: ``N // B`` batches
  when dropping, ``ceil(N / B)`` when not.
* Determinism: the train order is drawn from the SAME explicit CPU
  ``torch.Generator`` seeded from ``project.seed`` the legacy path uses
  (``DataPipelineStateMachine._train_sampler_generator``), so every epoch's
  order is reproducible from the project seed alone. The permutation STREAM
  differs from ``RandomSampler``'s consumption of that generator (same
  distribution, different sequence) — a disclosed, seed-reproducible loader
  change applied identically to every cell, so protocol fairness is
  unaffected. The permutation is always drawn on the CPU generator: portable
  across devices and immune to generator/device-mismatch rules.

The RNN cells (``*_rnn.yaml``, ``sequence_length=10``) keep the legacy
DataLoader path (padding collation and worker overlap); the pipeline
dispatches on ``data.sequence_length``/``data.stride``.

Pinned-memory contract
----------------------
Gathered batches are freshly allocated pageable host tensors, and the
trainer requests ``.to(device, non_blocking=True)`` on CUDA — a non-blocking
host-to-device copy only overlaps when the source is PINNED (pageable
sources copy synchronously; the legacy DataLoader pinned its batches in
exactly this situation). With ``pin_memory=True`` (the pipeline enables it
under the same gating as the legacy branch: config flag AND cuda target AND
CUDA available) each gathered batch is pinned before being yielded,
restoring the pinned-batch contract. Pinning is a pure copy (values
identical) and costs microseconds at these batch sizes (~30 KB per batch).
"""


from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor

from phaseforge.data.common.dataset import StateOnlyDataset


def _pin(tensor: Tensor) -> Tensor:
    """Pin one host tensor for asynchronous host-to-device copies.

    A module-level helper so the pinning route stays testable on CPU-only
    builds, where ``Tensor.pin_memory()`` raises (no accelerator present).
    """
    return tensor.pin_memory()


class InMemoryBatchLoader:
    """Drop-in batch source for single-step datasets (see module docstring).

    Iterating yields the same dict schema as the collator's single-step
    output: ``state (B,S)``, ``action (B,A)``, ``phase (B,)`` long,
    ``task_id``, ``trajectory_id``, ``trajectory_position`` — all ``(B,)``
    long. The class intentionally mirrors only the iteration surface of
    ``DataLoader`` (``__iter__``/``__len__``); loaders are consumed by plain
    iteration everywhere in this codebase.
    """

    def __init__(
        self,
        *,
        state: Tensor,
        action: Tensor,
        phase: Tensor,
        task_id: Tensor,
        trajectory_id: Tensor,
        trajectory_position: Tensor,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        generator: torch.Generator | None = None,
        pin_memory: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        n = state.size(0)
        lengths = {
            "state": n,
            "action": action.size(0),
            "phase": phase.numel(),
            "task_id": task_id.numel(),
            "trajectory_id": trajectory_id.numel(),
            "trajectory_position": trajectory_position.numel(),
        }
        mismatched = {k: v for k, v in lengths.items() if v != n}
        if mismatched:
            raise ValueError(f"flat fields disagree on row count: {mismatched}")
        if shuffle and generator is None:
            raise ValueError(
                "InMemoryBatchLoader(shuffle=True) requires an explicit "
                "torch.Generator so the sample order is reproducible from "
                "the project seed (the legacy RandomSampler contract)."
            )
        self.state = state
        self.action = action
        self.phase = phase
        self.task_id = task_id
        self.trajectory_id = trajectory_id
        self.trajectory_position = trajectory_position
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator
        # Enable ONLY under the legacy gating (cuda target AND CUDA
        # available): Tensor.pin_memory() raises on CPU-only builds.
        self.pin_memory = bool(pin_memory)
        self.dataset: StateOnlyDataset | None = None
        self._num_batches = n // self.batch_size if self.drop_last else (
            (n + self.batch_size - 1) // self.batch_size
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(
        cls,
        dataset: StateOnlyDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        generator: torch.Generator | None = None,
        pin_memory: bool = False,
    ) -> InMemoryBatchLoader:
        """Flatten a single-step :class:`StateOnlyDataset` into row tensors.

        The dataset's own construction (phase corruption included) has
        already run; this method only re-packs its rows, walking the SAME
        ``(traj_idx, start_t)`` index map the per-item path uses, so row
        order and content are identical by construction.
        """
        if dataset.sequence_length != 1 or dataset.stride != 1:
            raise ValueError(
                "InMemoryBatchLoader requires sequence_length=1 and stride=1 "
                f"(got {dataset.sequence_length}/{dataset.stride}); sequence "
                "configs must use the padding collator path."
            )
        index_map = dataset._index_map  # noqa: SLF001 - same-package flat re-packing
        trajectories = dataset.trajectories
        seq = dataset.sequence_length
        state = torch.cat(
            [trajectories[i]["state"][s : s + seq] for i, s in index_map], dim=0
        )
        action = torch.cat(
            [trajectories[i]["action"][s : s + seq] for i, s in index_map], dim=0
        )
        phase = torch.cat(
            [trajectories[i]["phase"][s : s + seq] for i, s in index_map], dim=0
        )
        task_id = torch.tensor(
            [int(trajectories[i]["task_id"]) for i, _ in index_map], dtype=torch.long
        )
        trajectory_id = torch.tensor([i for i, _ in index_map], dtype=torch.long)
        trajectory_position = torch.tensor([s for _, s in index_map], dtype=torch.long)
        loader = cls(
            state=state,
            action=action,
            phase=phase,
            task_id=task_id,
            trajectory_id=trajectory_id,
            trajectory_position=trajectory_position,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
            pin_memory=pin_memory,
        )
        # Keep the source dataset handle for tooling (mirrors DataLoader.dataset).
        loader.dataset = dataset
        return loader

    # ------------------------------------------------------------------
    # DataLoader iteration surface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        n = self.state.size(0)
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator)
        else:
            order = torch.arange(n)
        for start in range(0, self._num_batches * self.batch_size, self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = {
                "state": self.state[idx],
                "action": self.action[idx],
                "phase": self.phase[idx],
                "task_id": self.task_id[idx],
                "trajectory_id": self.trajectory_id[idx],
                "trajectory_position": self.trajectory_position[idx],
            }
            if self.pin_memory:
                batch = {key: _pin(value) for key, value in batch.items()}
            yield batch


__all__ = ["InMemoryBatchLoader"]
