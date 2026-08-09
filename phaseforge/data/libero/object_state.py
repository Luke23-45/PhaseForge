"""Decode per-object world poses from LIBERO HDF5 ``states`` arrays (numpy-only).

The ``states`` dataset at the demo root of every LIBERO HDF5 file is the
flattened MuJoCo simulation state, ``env.sim.get_state().flatten()``, i.e.
the concatenation of the model's ``qpos`` (``nq`` dims) and ``qvel`` (``nv``
dims). The dimension is task-specific (LIBERO issue #26, #71), so every task
needs its own index table that maps object bodies to their joint's ``qpos``
slice. Those tables are produced once per mirror revision by
``scripts/build_object_index.py`` (the patch-0 census) and validated there
against live robosuite observations (the B6 gate).

Why this decode is EXACT (train <-> eval parity, E3)
----------------------------------------------------
- Free joints: the ``qpos`` block ``[x, y, z, qw, qx, qy, qz]`` IS the body's
  world pose — MuJoCo free-joint coordinates are body-world coordinates, so
  the decoded pose equals ``sim.data.body_xpos`` / ``body_xquat`` up to
  float rounding, with no forward kinematics involved.
- Hinge joints (drawers, switches, doors): the body is rotated by the joint
  angle about a fixed world-space axis (its parent is static in every LIBERO
  scene). The census captures the world-frame hinge constants once; the
  decode is a closed-form screw rotation, mathematically identical to
  MuJoCo's own FK, so it still matches the live observation exactly.
- Slide joints (sliding doors): pure translation along a fixed world-space
  axis, also decoded closed-form from the census constants.
- Fixed bodies: constant pose, copied from the table.

The decoded object block is concatenated AFTER the 23-dim proprioceptive
block, in table order (names sorted), zero-padded to ``k_slots``. When the
index's ``include_mask`` is True it is followed by a binary ``mask`` of
length ``k_slots`` (1.0 = slot filled); when False the mask is omitted
entirely (``decode`` returns a ``(T, 0)`` mask, a no-op on concatenation).
The mask dims are excluded from normalization by the pipeline
(``normalizer.ignore_dims``). The eval-side env consumes the same flag, so
train <-> eval state layouts stay identical by construction.

Layout::

    state = [ proprio (23) | object_block (k_slots * 7) | mask (k_slots) ]

Quaternion convention: w-first ``(w, x, y, z)``, Hamilton product — the
MuJoCo / robosuite convention used by ``body_xquat`` and free-joint qpos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Dimensionality of one decoded object slot: world pos (3) + quat (4).
#: Mirrors the robosuite ``{name}_pos`` / ``{name}_quat`` observables.
POS_QUAT_DIM = 7

#: Joint types understood by the decoder.
JOINT_FREE = "free"    # qpos block is the body world pose directly
JOINT_HINGE = "hinge"  # qpos is 1 scalar angle; pose via screw rotation
JOINT_SLIDE = "slide"  # qpos is 1 scalar distance; pose via axis translation
JOINT_FIXED = "fixed"  # static pose; no qpos entry

_JOINT_TYPES = (JOINT_FREE, JOINT_HINGE, JOINT_SLIDE, JOINT_FIXED)

#: Suffix the LIBERO mirror appends to every task file (``foo_demo.hdf5``).
#: Verified: 100% of files in both libero_90/ and libero_10/ end with this
#: (see the download script's manifest). The canonical task name used as
#: the ObjectIndex key is the benchmark name — the stem WITHOUT this
#: suffix — because the ingest side consumes HDF5 stems while the eval
#: side consumes benchmark task names; the chokepoint in :meth:`table`
#: accepts both forms.
DEMO_SUFFIX = "_demo"


# ---------------------------------------------------------------------------
# Quaternion helpers (w-first, MuJoCo convention)
# ---------------------------------------------------------------------------


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q2`` (w-first), vectorized over the last axis.

    Matches MuJoCo's ``quatMul`` sign conventions.
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        ],
        axis=-1,
    )


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Quaternion (w-first) encoding of a rotation about a unit ``axis``.

    ``axis`` has shape (3,) or (..., 3); ``angle`` shape (...,).
    """
    half = np.asarray(angle, dtype=np.float64) * 0.5
    s = np.sin(half)
    ax = np.asarray(axis, dtype=np.float64)
    q = np.empty(half.shape + (4,), dtype=np.float64)
    q[..., 0] = np.cos(half)
    q[..., 1] = ax[..., 0] * s
    q[..., 2] = ax[..., 1] * s
    q[..., 3] = ax[..., 2] * s
    return q


def rotate_vec_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (..., 3) by quaternions ``q`` (..., 4), w-first."""
    u = q[..., 1:4]                       # vector part
    s = q[..., 0:1]                       # scalar part, (..., 1)
    cross = np.cross(u, v)
    return v + 2.0 * (s * cross + np.cross(u, cross))


def _canonical_task_name(task_name: str) -> str:
    """Strip the mirror's ``_demo`` filename suffix when present.

    Idempotent: benchmark task names never end in ``_demo``, so canonical
    queries pass through unchanged.
    """
    if task_name.endswith(DEMO_SUFFIX):
        return task_name[: -len(DEMO_SUFFIX)]
    return task_name


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Index table data model
# ---------------------------------------------------------------------------


@dataclass
class ObjectEntry:
    """Decode recipe for a single object body of a task.

    Attributes:
        name:       Object name, e.g. ``"book"``. Also the robosuite
                    observable key stem at eval (``{name}_pos``).
        joint_type: One of :data:`JOINT_FREE`, :data:`JOINT_HINGE`,
                    :data:`JOINT_FIXED`.
        qpos_start: Index of this joint's first ``qpos`` entry (into the
                    qpos half of ``states``).
        qpos_len:   7 for free joints (pos + quat), 1 for hinge/slide
                    (scalar), 0 for fixed.
        anchor_world: (3,) world-space point on the hinge axis (hinge only).
        axis_world:   (3,) world-space unit hinge axis / slide direction
                      (hinge/slide only).
        rest_pos:     (3,) body world position at joint angle 0 (hinge/fixed).
        rest_quat:    (4,) body world quaternion at joint angle 0 (w-first).
    """

    name: str
    joint_type: str
    qpos_start: int = 0
    qpos_len: int = 0
    anchor_world: np.ndarray | None = None
    axis_world: np.ndarray | None = None
    rest_pos: np.ndarray | None = None
    rest_quat: np.ndarray | None = None


@dataclass
class TaskObjectTable:
    """Decode table for one task.

    Attributes:
        task_name:  Canonical task name — the benchmark name, i.e. the
                    HDF5 filename stem minus the ``_demo`` suffix (e.g.
                    ``"KITCHEN_SCENE1_open_drawer"``).
        nq:         Number of ``qpos`` entries in the task's ``states``
                    array (the qpos half length; qvel half = total - nq).
        objects:    Object entries in fixed slot order (names sorted);
                    slot index == list index, zero-padded to ``k_slots``.
    """

    task_name: str
    nq: int
    objects: list[ObjectEntry] = field(default_factory=list)


class ObjectIndex:
    """Per-task decode tables, loaded once from the census-built JSON.

    The JSON is produced by ``phaseforge.data.scripts.build_object_index``
    (patch 0, run on the machine with the mirror + LIBERO installed) and lives
    at ``{data_root}/raw/libero/object_index.json`` by default (see
    :func:`phaseforge.data.paths.libero_object_index_path`).

    Both the ingest side (decode from ``states``) and the eval side (read
    ``{name}_pos`` / ``{name}_quat`` from live robosuite obs) consume the
    SAME table, so train <-> eval object selection and ordering match by
    construction.
    """

    def __init__(
        self,
        tasks: dict[str, TaskObjectTable],
        k_slots: int,
        dim_per_object: int = POS_QUAT_DIM,
        include_mask: bool = True,
    ) -> None:
        if k_slots <= 0:
            raise ValueError(f"k_slots must be >= 1, got {k_slots}")
        if dim_per_object != POS_QUAT_DIM:
            raise ValueError(
                f"dim_per_object must be {POS_QUAT_DIM} (pos 3 + quat 4), "
                f"got {dim_per_object}"
            )
        if not tasks:
            raise ValueError("ObjectIndex requires at least one task table")
        self.tasks = tasks
        self.k_slots = int(k_slots)
        self.dim_per_object = int(dim_per_object)
        self.include_mask = bool(include_mask)
        self._validate_tables()

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------

    def _validate_tables(self) -> None:
        for task_name, table in self.tasks.items():
            if table.task_name != task_name:
                raise ValueError(
                    f"Table key {task_name!r} does not match table.task_name "
                    f"{table.task_name!r}"
                )
            if table.nq <= 0:
                raise ValueError(f"Task {task_name}: nq must be >= 1, got {table.nq}")
            seen: set[str] = set()
            for entry in table.objects:
                if entry.name in seen:
                    raise ValueError(
                        f"Task {task_name}: duplicate object {entry.name!r}"
                    )
                seen.add(entry.name)
                if entry.joint_type not in _JOINT_TYPES:
                    raise ValueError(
                        f"Task {task_name}, object {entry.name!r}: unknown "
                        f"joint_type {entry.joint_type!r}"
                    )
                if entry.joint_type == JOINT_FREE and entry.qpos_len != POS_QUAT_DIM:
                    raise ValueError(
                        f"Task {task_name}, object {entry.name!r}: free joint "
                        f"must have qpos_len={POS_QUAT_DIM}"
                    )
                if entry.joint_type in (JOINT_HINGE, JOINT_SLIDE) and entry.qpos_len != 1:
                    raise ValueError(
                        f"Task {task_name}, object {entry.name!r}: "
                        f"{entry.joint_type} joint must have qpos_len=1"
                    )
                if entry.joint_type in (JOINT_HINGE, JOINT_SLIDE, JOINT_FIXED):
                    for attr, need in (
                        ("anchor_world", entry.joint_type == JOINT_HINGE),
                        ("axis_world", entry.joint_type in (JOINT_HINGE, JOINT_SLIDE)),
                        ("rest_pos", True),
                        ("rest_quat", True),
                    ):
                        value = getattr(entry, attr)
                        if need and value is None:
                            raise ValueError(
                                f"Task {task_name}, object {entry.name!r}: "
                                f"missing {attr} for {entry.joint_type} joint"
                            )
                if entry.joint_type == JOINT_FREE:
                    if entry.qpos_start < 0 or entry.qpos_start + entry.qpos_len > table.nq:
                        raise ValueError(
                            f"Task {task_name}, object {entry.name!r}: qpos "
                            f"slice [{entry.qpos_start}, "
                            f"{entry.qpos_start + entry.qpos_len}) out of "
                            f"range for nq={table.nq}"
                        )

    # ------------------------------------------------------------------
    # JSON (de)serialization
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> ObjectIndex:
        """Load an index from the census-produced JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Object index not found: {path}. Build it first with: "
                "python -m phaseforge.data.scripts.build_object_index"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Object index {path} is not valid JSON: {exc}") from exc

        if raw.get("version") != 1:
            raise ValueError(
                f"Object index {path}: unsupported version {raw.get('version')!r} "
                "(expected 1)"
            )
        k_slots = int(raw.get("k_slots", -1))
        dim_per_object = int(raw.get("dim_per_object", POS_QUAT_DIM))
        include_mask = bool(raw.get("include_mask", True))

        tasks: dict[str, TaskObjectTable] = {}
        raw_tasks = raw.get("tasks", {})
        for task_name, raw_table in raw_tasks.items():
            entries: list[ObjectEntry] = []
            for raw_obj in raw_table.get("objects", []):
                raw_rest_quat = _as_float_array(raw_obj.get("rest_quat"), 4)
                entries.append(
                    ObjectEntry(
                        name=str(raw_obj["name"]),
                        joint_type=str(raw_obj["joint_type"]),
                        qpos_start=int(raw_obj.get("qpos_start", 0)),
                        qpos_len=int(raw_obj.get("qpos_len", 0)),
                        anchor_world=_as_float_array(raw_obj.get("anchor_world"), 3),
                        axis_world=_as_float_array(raw_obj.get("axis_world"), 3),
                        rest_pos=_as_float_array(raw_obj.get("rest_pos"), 3),
                        rest_quat=(
                            _normalize_quat(raw_rest_quat)
                            if raw_rest_quat is not None
                            else None
                        ),
                    )
                )
            tasks[str(task_name)] = TaskObjectTable(
                task_name=str(task_name),
                nq=int(raw_table.get("nq", -1)),
                objects=entries,
            )

        index = cls(tasks=tasks, k_slots=k_slots, dim_per_object=dim_per_object,
                    include_mask=include_mask)
        logger.info(
            "Loaded object index from %s: %d tasks, k_slots=%d",
            path, len(index.tasks), index.k_slots,
        )
        return index

    def save(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> None:
        """Serialize the index back to JSON (used by the census script)."""
        payload: dict[str, Any] = {
            "version": 1,
            "k_slots": self.k_slots,
            "dim_per_object": self.dim_per_object,
            "include_mask": self.include_mask,
            "tasks": {},
        }
        if extra_meta:
            payload.update(extra_meta)
        for task_name, table in self.tasks.items():
            payload["tasks"][task_name] = {
                "nq": table.nq,
                "objects": [
                    {
                        "name": e.name,
                        "joint_type": e.joint_type,
                        "qpos_start": e.qpos_start,
                        "qpos_len": e.qpos_len,
                        **(
                            {
                                "anchor_world": e.anchor_world.tolist(),
                            }
                            if e.anchor_world is not None
                            else {}
                        ),
                        **(
                            {
                                "axis_world": e.axis_world.tolist(),
                            }
                            if e.axis_world is not None
                            else {}
                        ),
                        **(
                            {
                                "rest_pos": e.rest_pos.tolist(),
                                "rest_quat": e.rest_quat.tolist(),
                            }
                            if e.rest_pos is not None
                            else {}
                        ),
                    }
                    for e in table.objects
                ],
            }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def table(self, task_name: str) -> TaskObjectTable:
        """Return the decode table for a task, raising a helpful error.

        Accepts either the canonical task name or the HDF5 filename stem
        (with the ``_demo`` suffix) — both forms appear in the codebase
        (the ingest side keys by stem, the eval side by benchmark name),
        and this chokepoint canonicalizes them so the two can never miss.
        """
        key = _canonical_task_name(task_name)
        try:
            return self.tasks[key]
        except KeyError as exc:
            known = sorted(self.tasks)[:5]
            raise KeyError(
                f"Task {task_name!r} not found in the object index "
                f"({len(self.tasks)} tasks total, e.g. {known}). The index "
                "must be rebuilt for the exact suite revision in use."
            ) from exc

    def object_names(self, task_name: str) -> list[str]:
        """Object names in slot order for a task (eval-side reading)."""
        return [e.name for e in self.table(task_name).objects]

    def n_objects(self, task_name: str) -> int:
        return len(self.table(task_name).objects)

    @property
    def state_block_size(self) -> int:
        """Size of the object-state portion of the state vector (block+mask)."""
        block = self.k_slots * self.dim_per_object
        return block + (self.k_slots if self.include_mask else 0)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, task_name: str, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode the object-state block from a task's ``states`` array.

        Args:
            task_name: Canonical task name (benchmark name; the ``_demo``
                       filename suffix is stripped automatically).
            states:    ``(T, S)`` array — the demo's ``states`` dataset
                       (flattened sim state: ``qpos`` then ``qvel``).

        Returns:
            ``(object_block, mask)`` with shapes ``(T, k_slots*7)`` and
            ``(T, k_slots)`` — or ``(T, 0)`` when ``include_mask`` is False —
            both float32. Empty slots are zero; mask is 1.0 for filled
            slots, 0.0 otherwise. Concatenating the ``(T, 0)`` mask is a
            no-op, so callers can append it unconditionally.
        """
        table = self.table(task_name)
        states = np.asarray(states)
        if states.ndim != 2:
            raise ValueError(
                f"Task {task_name}: states must be (T, S), got shape {states.shape}"
            )
        if states.shape[-1] < table.nq:
            raise ValueError(
                f"Task {task_name}: states has {states.shape[-1]} dims but the "
                f"table expects at least nq={table.nq} (qpos half)."
            )
        n_objects = len(table.objects)
        if n_objects > self.k_slots:
            raise ValueError(
                f"Task {task_name}: {n_objects} objects but k_slots={self.k_slots}. "
                "Raise object_state.k_slots and rebuild the cache (never silently "
                "truncate object state)."
            )

        T = states.shape[0]
        qpos = states[:, : table.nq]
        block = np.zeros((T, self.k_slots * self.dim_per_object), dtype=np.float32)
        mask = np.zeros(
            (T, self.k_slots if self.include_mask else 0), dtype=np.float32
        )

        for slot, entry in enumerate(table.objects):
            start = slot * self.dim_per_object
            pose = self._decode_entry(entry, qpos, T)
            block[:, start : start + self.dim_per_object] = pose
            if self.include_mask:
                mask[:, slot] = 1.0

        return block, mask

    def _decode_entry(self, entry: ObjectEntry, qpos: np.ndarray, T: int) -> np.ndarray:
        """Decode one object's ``(T, 7)`` world pose [pos(3), quat(4)]."""
        if entry.joint_type == JOINT_FREE:
            return qpos[:, entry.qpos_start : entry.qpos_start + POS_QUAT_DIM].astype(
                np.float32
            )

        if entry.joint_type == JOINT_FIXED:
            pose = np.empty((T, POS_QUAT_DIM), dtype=np.float32)
            pose[:, :3] = entry.rest_pos.astype(np.float32)
            pose[:, 3:] = entry.rest_quat.astype(np.float32)
            return pose

        if entry.joint_type == JOINT_SLIDE:
            # Slide: body translates along the census-captured world-space
            # axis by the qpos distance (measured from rest at qpos=0).
            axis = entry.axis_world.astype(np.float64)
            distance = qpos[:, entry.qpos_start].astype(np.float64)[..., None]
            pose = np.empty((T, POS_QUAT_DIM), dtype=np.float32)
            pose[:, :3] = (
                entry.rest_pos.astype(np.float64) + axis * distance
            ).astype(np.float32)
            pose[:, 3:] = entry.rest_quat.astype(np.float32)
            return pose

        # Hinge: closed-form screw rotation about the census-captured
        # world-space axis (parent is static in every LIBERO scene).
        angle = qpos[:, entry.qpos_start].astype(np.float64)
        axis = entry.axis_world.astype(np.float64)
        anchor = entry.anchor_world.astype(np.float64)
        rest_pos = entry.rest_pos.astype(np.float64)
        rest_quat = entry.rest_quat.astype(np.float64)

        quat = quat_from_axis_angle(axis, angle)          # (T, 4)
        offset = rest_pos - anchor                        # (3,)
        pos = anchor + rotate_vec_by_quat(np.broadcast_to(offset, (T, 3)), quat)
        ori = _normalize_quat(quat_multiply(quat, np.broadcast_to(rest_quat, (T, 4))))

        pose = np.empty((T, POS_QUAT_DIM), dtype=np.float32)
        pose[:, :3] = pos.astype(np.float32)
        pose[:, 3:] = ori.astype(np.float32)
        return pose


def _as_float_array(value: Any, size: int) -> np.ndarray | None:
    """Convert a JSON list to a float64 numpy array, or None."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != size:
        raise ValueError(f"Expected {size} floats, got {arr.shape[0]}")
    return arr
