"""Method/task registry — derived from the frozen protocol manifests.

The runner's ``load_protocol`` is the single source of truth; this module adds
the paper-facing display layer (names, ordering) on top, so the analysis can
never disagree with what actually ran.
"""

from __future__ import annotations

from functools import cache

from phaseforge.runner.protocol import Method, Protocol, load_protocol
from studies.analysis.common.config import namespace_manifest

#: Paper-facing display names (figures_tables_plan.md style contract).
DISPLAY_NAMES: dict[str, str] = {
    "phaseforge": "PhaseForge",
    "bc": "BC",
    "bc_large": "BC-Large",
    "bc_rnn": "BC-RNN",
    "bc_robot_only": "BC Robot-Only",
    "scratch_moe": "Scratch MoE",
    "warmstart_moe": "Warm-Start MoE",
    "phase_pretrain_random_router": "PP Random-Router",
    "plain_encoder_phase_bootstrap": "PE Phase-Bootstrap",
    "teacher_forced": "Teacher-Forced",
    # Router Init Family (Group A)
    "pf_spherical_kmeans": "Spherical K-Means",
    "pf_kmeans": "Euclidean K-Means",
    "pf_phase_head": "Phase-Head Directions",
    "pf_random_random": "Random Router (H1)",
    "pf_centroid_random": "Centroid + Rand Exp",
    # Drop-rate Sweep (Group B)
    "pf_drop00": "Drop 0% (Full Warm)",
    "pf_drop25": "Drop 25%",
    "pf_drop50": "Drop 50% (PhaseForge)",
    "pf_drop75": "Drop 75%",
    "pf_drop100": "Drop 100% (Scratch Exp)",
    "pf_full_warm": "Full Warm-Start",
    "pf_one_warm_plus_random": "One-Warm + 5-Rand",
    # Phase Quality & Supervision (Group C)
    "pf_corrupt_25": "25% Phase Corruption",
    "pf_corrupt_50": "50% Phase Corruption",
    "pf_shuffle_control": "100% Phase Shuffle",
    # Expert Capacity & Encoder (Groups D & E)
    "pf_k3": "K=3 Experts",
    "pf_k12": "K=12 Experts",
    "pf_spherical": "Spherical Latent Space",
    "pf_ft": "Fine-Tuned Encoder",
}

#: T1/T2 row order: proposed method first, then floors, controls, diagnostics.
MATRIX_ORDER: tuple[str, ...] = (
    "phaseforge",
    "bc",
    "bc_large",
    "bc_rnn",
    "bc_robot_only",
    "scratch_moe",
    "warmstart_moe",
    "phase_pretrain_random_router",
    "plain_encoder_phase_bootstrap",
    "teacher_forced",
)

TASK_ORDER: tuple[str, ...] = ("Lift", "Can", "Square", "ToolHang", "Transport")


@cache
def protocol(namespace: str) -> Protocol:
    return load_protocol(namespace_manifest(namespace))


def methods(namespace: str) -> tuple[Method, ...]:
    return protocol(namespace).methods


def seeds(namespace: str) -> tuple[int, ...]:
    return protocol(namespace).seeds


def matrix_method_names() -> tuple[str, ...]:
    """The 10 five-task method names in canonical paper order."""
    present = {m.name for m in methods("final")}
    return tuple(name for name in MATRIX_ORDER if name in present)


def tasks() -> tuple[str, ...]:
    present = {m.task for m in methods("final") if m.task is not None}
    return tuple(t for t in TASK_ORDER if t in present)


def ablation_method_names() -> tuple[str, ...]:
    """Ablation-only cells (not part of the ten-method matrix), manifest order."""
    matrix = set(matrix_method_names())
    seen: list[str] = []
    for m in methods("ablation"):
        if m.name not in matrix and m.name not in seen:
            seen.append(m.name)
    return tuple(seen)


def display_name(method_name: str) -> str:
    return DISPLAY_NAMES.get(method_name, method_name)


def experiment_id(namespace: str, method_name: str) -> str | None:
    for m in methods(namespace):
        if m.name == method_name:
            return m.experiment_id
    return None


def expected_cells(namespace: str) -> tuple[tuple[str | None, str], ...]:
    """Every (task, method-name) cell the namespace must contain."""
    return tuple({(m.task, m.name) for m in methods(namespace)})
