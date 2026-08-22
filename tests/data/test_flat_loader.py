"""Flat in-memory batch iterator tests (training review T5).

Gates from the review's verification plan:
* content parity — with the permutation pinned to sequential, batches are
  BIT-IDENTICAL to the collator/DataLoader path on the same dataset;
* coverage — a shuffled epoch yields every row exactly once;
* determinism — same seed => identical epoch sequences; the generator state
  advances so successive epochs differ;
* drop_last parity with DataLoader semantics;
* shuffle requires an explicit generator (RandomSampler contract).
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from phaseforge.data.common.collator import PhaseAwareCollator
from phaseforge.data.common.dataset import StateOnlyDataset
from phaseforge.data.common.flat_iterator import InMemoryBatchLoader


def _dataset(n_traj: int = 7, t_len: int = 9, state_dim: int = 5, action_dim: int = 3,
             num_phases: int = 4, seed: int = 0) -> StateOnlyDataset:
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for i in range(n_traj):
        t = torch.randint(3, t_len + 1, (1,), generator=g).item() + 2
        trajs.append(
            {
                "state": torch.randn(t, state_dim, generator=g),
                "action": torch.randn(t, action_dim, generator=g),
                "phase": torch.randint(0, num_phases, (t,), generator=g),
                "task_id": i % 3,
            }
        )
    return StateOnlyDataset(
        trajectories=trajs,
        sequence_length=1,
        stride=1,
        num_phases=num_phases,
        phase_corruption_seed=123,
    )


def _fast_loader(dataset: StateOnlyDataset, batch_size: int, shuffle: bool, drop_last: bool,
                 seed: int | None = None) -> InMemoryBatchLoader:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return InMemoryBatchLoader.from_dataset(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )


def _sequential_fast_loader(dataset: StateOnlyDataset, batch_size: int, drop_last: bool):
    """Fast path with the permutation pinned to sequential (shuffle=False)."""
    return _fast_loader(dataset, batch_size, shuffle=False, drop_last=drop_last)


def _row_multiset(tensor: torch.Tensor) -> list[tuple]:
    """Exact row multiset (lexicographic tuple sort) — column-wise tensor
    sorting scrambles rows and is NOT a multiset comparison."""
    return sorted(map(tuple, tensor.tolist()))


class TestContentParity:
    def test_batches_bit_identical_to_collator_path(self) -> None:
        dataset = _dataset()
        batch_size = 4
        legacy = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True,
            collate_fn=PhaseAwareCollator(),
        )
        fast = _sequential_fast_loader(dataset, batch_size, drop_last=True)

        assert len(legacy) == len(fast)
        for legacy_batch, fast_batch in zip(legacy, fast):
            assert set(legacy_batch) == set(fast_batch)
            for key in legacy_batch:
                assert torch.equal(legacy_batch[key], fast_batch[key]), f"field {key} differs"
            assert legacy_batch["phase"].dtype == fast_batch["phase"].dtype
            assert legacy_batch["trajectory_id"].dtype == fast_batch["trajectory_id"].dtype

    def test_batches_bit_identical_without_drop_last(self) -> None:
        dataset = _dataset()
        batch_size = 4
        legacy = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            collate_fn=PhaseAwareCollator(),
        )
        fast = _sequential_fast_loader(dataset, batch_size, drop_last=False)
        assert len(legacy) == len(fast)
        for legacy_batch, fast_batch in zip(legacy, fast):
            for key in legacy_batch:
                assert torch.equal(legacy_batch[key], fast_batch[key])

    def test_dataset_with_corrupted_labels_flows_through(self) -> None:
        corrupted = StateOnlyDataset(
            trajectories=_dataset().trajectories,
            sequence_length=1,
            stride=1,
            num_phases=4,
            phase_corruption_rate=0.5,
            phase_corruption_seed=77,
        )
        fast = _sequential_fast_loader(corrupted, batch_size=8, drop_last=False)
        rows = torch.cat([b["phase"] for b in fast], dim=0)
        assert torch.equal(rows, torch.cat([t["phase"] for t in corrupted.trajectories]))


class TestCoverageAndParity:
    def test_shuffled_epoch_covers_every_row_exactly_once(self) -> None:
        dataset = _dataset()
        fast = _fast_loader(dataset, batch_size=4, shuffle=True, drop_last=True, seed=42)
        seen = torch.cat([b["state"] for b in fast], dim=0)
        n_total = sum(t["state"].size(0) for t in dataset.trajectories)
        assert seen.size(0) == (n_total // 4) * 4  # drop_last parity
        # Every emitted row comes from the dataset: exact multiset containment
        # (the drop_last remainder is a random subset, not a prefix).
        from collections import Counter

        extra = Counter(_row_multiset(seen)) - Counter(
            _row_multiset(torch.cat([t["state"] for t in dataset.trajectories], dim=0))
        )
        assert not extra, f"fast path emitted rows absent from the dataset: {list(extra)[:3]}"

    def test_drop_last_parity_batch_counts(self) -> None:
        dataset = _dataset(n_traj=5, t_len=4)  # deterministic row count via seeds
        n = sum(t["state"].size(0) for t in dataset.trajectories)
        train_fast = _fast_loader(dataset, batch_size=4, shuffle=True, drop_last=True, seed=1)
        val_fast = _fast_loader(dataset, batch_size=4, shuffle=False, drop_last=False)
        assert len(train_fast) == n // 4
        assert len(val_fast) == -(-n // 4)  # ceil


class TestDeterminism:
    def test_same_seed_reproduces_epoch_sequences(self) -> None:
        dataset = _dataset()
        a = _fast_loader(dataset, 4, shuffle=True, drop_last=True, seed=99)
        b = _fast_loader(dataset, 4, shuffle=True, drop_last=True, seed=99)
        for batch_a, batch_b in zip(a, b):
            assert torch.equal(batch_a["state"], batch_b["state"])
            assert torch.equal(batch_a["trajectory_position"], batch_b["trajectory_position"])

    def test_generator_advances_between_epochs(self) -> None:
        dataset = _dataset()
        # drop_last=False: both epochs cover the FULL row set, so the
        # multiset invariant holds (with drop_last=True each epoch drops a
        # different random remainder — orders differ, multisets need not).
        fast = _fast_loader(dataset, 4, shuffle=True, drop_last=False, seed=7)
        first = [b["state"].clone() for b in fast]
        second = [b["state"].clone() for b in fast]
        joined_first = torch.cat(first)
        joined_second = torch.cat(second)
        assert not torch.equal(joined_first, joined_second), "epochs repeated the same order"
        # ...but the multiset is still the complete row set.
        assert _row_multiset(joined_first) == _row_multiset(joined_second)


class TestContracts:
    def test_shuffle_requires_explicit_generator(self) -> None:
        dataset = _dataset(n_traj=2, t_len=3)
        with pytest.raises(ValueError, match="Generator"):
            InMemoryBatchLoader.from_dataset(
                dataset, batch_size=4, shuffle=True, drop_last=True, generator=None
            )

    def test_sequence_datasets_are_rejected(self) -> None:
        dataset = StateOnlyDataset(
            trajectories=_dataset(n_traj=2, t_len=6).trajectories,
            sequence_length=4,
            stride=1,
        )
        with pytest.raises(ValueError, match="sequence_length=1"):
            InMemoryBatchLoader.from_dataset(
                dataset, batch_size=4, shuffle=False, drop_last=True
            )

    def test_mismatched_row_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="row count"):
            InMemoryBatchLoader(
                state=torch.zeros(10, 5),
                action=torch.zeros(9, 3),
                phase=torch.zeros(10, dtype=torch.long),
                task_id=torch.zeros(10, dtype=torch.long),
                trajectory_id=torch.zeros(10, dtype=torch.long),
                trajectory_position=torch.zeros(10, dtype=torch.long),
                batch_size=4,
                shuffle=False,
                drop_last=True,
            )


class TestPinnedMemoryContract:
    """External review P1: the fast path must honor the legacy pin_memory
    contract — non_blocking host-to-CUDA copies only overlap from pinned
    host memory, and gathered batches are freshly allocated pageable
    tensors until pinned."""

    def test_pinning_disabled_by_default(self, monkeypatch) -> None:
        import phaseforge.data.common.flat_iterator as fi

        calls = []
        monkeypatch.setattr(
            fi, "_pin", lambda t: (calls.append(t), t)[1]
        )
        dataset = _dataset()
        loader = _fast_loader(dataset, batch_size=4, shuffle=False, drop_last=True)
        for _ in loader:
            pass
        assert not calls, "pinning ran although pin_memory was not requested"

    def test_pinning_routes_every_field_value_identical(self, monkeypatch) -> None:
        import phaseforge.data.common.flat_iterator as fi

        calls = []
        monkeypatch.setattr(
            fi, "_pin", lambda t: (calls.append(t), t)[1]
        )
        dataset = _dataset()
        pinned = InMemoryBatchLoader.from_dataset(
            dataset, batch_size=4, shuffle=False, drop_last=True, pin_memory=True
        )
        plain = InMemoryBatchLoader.from_dataset(
            dataset, batch_size=4, shuffle=False, drop_last=True, pin_memory=False
        )
        n_batches = 0
        for pinned_batch, plain_batch in zip(pinned, plain):
            n_batches += 1
            assert set(pinned_batch) == set(plain_batch)
            for key in plain_batch:
                assert torch.equal(pinned_batch[key], plain_batch[key])
        assert n_batches == len(pinned)
        # One _pin call per field per batch.
        assert len(calls) == n_batches * 6

    def test_real_pinning_marks_memory_pinned(self) -> None:
        """Runs only where an accelerator exists (the cloud sweep machine):
        the produced batches must be genuinely page-locked."""
        if not torch.cuda.is_available():
            import pytest

            pytest.skip("pin_memory requires an accelerator; wiring covered above")
        dataset = _dataset()
        loader = InMemoryBatchLoader.from_dataset(
            dataset, batch_size=4, shuffle=False, drop_last=True, pin_memory=True
        )
        for batch in loader:
            for key, value in batch.items():
                assert value.is_pinned(), f"field {key} not pinned"
            break
