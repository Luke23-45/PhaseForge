"""
CPU-optimized smoke & diagnostic tests for PhaseForge Can/Square underperformance.
No GPU required. Reads raw HDF5 + processed cache + training curves + eval results.
Designed to validate every hypothesis for the failure: phaseforge 0.30/0.13 vs baselines 0.48/0.21.

Run: pytest tests/debug/test_phaseforge_can_square_cpu.py -v -s
or: py tests/debug/test_phaseforge_can_square_cpu.py
"""
from __future__ import annotations
import json
import h5py
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict
import yaml

# ---------------------------------------------------------------------------
# Helpers: raw data paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/data/raw/robomimic")
CACHE_ROOT = Path(r"C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/data/processed/cache")
OUTPUTS_FINAL = Path(r"C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/outputs_final")
PHASEFORGE_STAGE1 = OUTPUTS_FINAL / "phaseforge" / "stage1"
PHASEFORGE_STAGE2 = OUTPUTS_FINAL / "phaseforge" / "stage2"
EVAL_BASE = OUTPUTS_FINAL / "eval"

TASKS = ["Can", "Square", "Lift"]
TASK_TO_HDF5 = {
    "Can": DATA_ROOT / "can" / "low_dim_v15.hdf5",
    "Lift": DATA_ROOT / "lift" / "low_dim_v15.hdf5",
    "Square": DATA_ROOT / "square" / "low_dim_v15.hdf5",
}

# Import labeler (CPU)
from phaseforge.data.robomimic.phase_labeler import RuleBasedPhaseLabeler

# ---------------------------------------------------------------------------
# TEST 1: Phase distribution per task via RuleBasedPhaseLabeler (CPU)
# ---------------------------------------------------------------------------
def test_phase_distribution_per_task():
    """
    Hypothesis A1: Can/Square phase distribution heavily skewed vs Lift -> load imbalance.
    We compute offline labels for first N demos per task and report histogram.
    Expected: Can/Square have rarer transport/place phases due to object dynamics.
    """
    print("\n=== TEST 1: Phase distribution per task ===")
    labeler = RuleBasedPhaseLabeler()  # default 6 phases
    results = {}
    for task in TASKS:
        hdf5_path = TASK_TO_HDF5[task]
        assert hdf5_path.exists(), f"Missing {hdf5_path}"
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            demos = [k for k in data.keys() if k.startswith("demo_")][:10]  # CPU: sample 10 demos
            all_phases = []
            for demo_key in demos:
                demo = data[demo_key]
                obs = demo["obs"]
                # Reconstruct state as ingestion does: robot_eef_pos (3) + quat (4) + gripper (2) + object (14/10)
                # Use same slices as config: state_dim 23 for Can/Square, 19 for Lift
                # For phase labeling, only eef_pos 0:3 and gripper 7:9 matter, so we can build dummy state with those positions
                # Simpler: read needed keys directly from obs and concatenate
                eef_pos = np.asarray(obs["robot0_eef_pos"][:], dtype=np.float32)  # T,3
                eef_quat = np.asarray(obs["robot0_eef_quat"][:], dtype=np.float32)  # T,4
                gripper_qpos = np.asarray(obs["robot0_gripper_qpos"][:], dtype=np.float32)  # T,2
                obj = np.asarray(obs["object"][:], dtype=np.float32)  # T, 14 or 10
                state = np.concatenate([eef_pos, eef_quat, gripper_qpos, obj], axis=1)  # T, D
                phases = labeler.label({"state": state})
                all_phases.extend(phases.tolist())
            cnt = Counter(all_phases)
            total = len(all_phases)
            dist = {int(k): round(v/total, 3) for k, v in sorted(cnt.items())}
            imbalance = max(cnt.values()) / max(1, min(cnt.values())) if cnt else 0
            print(f" {task:8} total_steps={total} dist={dist} imbalance_max/min={imbalance:.2f} unique_phases={sorted(cnt.keys())}")
            results[task] = dist
    # Assertion for smoke: all tasks should have 6 phases present, but check skew
    for task, dist in results.items():
        assert len(dist) == 6, f"{task} missing phases: {dist}"
        # No phase should dominate >60% (heuristic for imbalance)
        max_frac = max(dist.values())
        if max_frac > 0.6:
            print(f"  WARNING {task} highly skewed phase {max_frac:.2f} may cause MoE under-utilization")

# ---------------------------------------------------------------------------
# TEST 2: Phase labeler sensitivity (CPU)
# ---------------------------------------------------------------------------
def test_phase_labeler_sensitivity():
    """
    Hypothesis A2/A3/A4: velocity_threshold, min_duration, median filter mis-tuned for Square.
    We take one Can demo and vary params to see phase sequence changes.
    """
    print("\n=== TEST 2: Phase labeler sensitivity ===")
    task = "Square"
    hdf5_path = TASK_TO_HDF5[task]
    with h5py.File(hdf5_path, "r") as f:
        demo_key = [k for k in f["data"].keys() if k.startswith("demo_")][0]
        obs = f["data"][demo_key]["obs"]
        eef_pos = np.asarray(obs["robot0_eef_pos"][:], dtype=np.float32)
        eef_quat = np.asarray(obs["robot0_eef_quat"][:], dtype=np.float32)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"][:], dtype=np.float32)
        obj = np.asarray(obs["object"][:], dtype=np.float32)
        state = np.concatenate([eef_pos, eef_quat, gripper_qpos, obj], axis=1)
    base = RuleBasedPhaseLabeler().label({"state": state})
    print(f" base phases for {task} demo {demo_key} len={len(base)} unique {np.unique(base)}")
    for vel in [0.005, 0.01, 0.02]:
        lab = RuleBasedPhaseLabeler(eef_velocity_threshold=vel).label({"state": state})
        diff = np.mean(base != lab)
        print(f"  vel_thr={vel:.3f} diff_vs_base={diff:.3f} phases {np.unique(lab)}")
    for filt in [3, 7, 11]:
        lab = RuleBasedPhaseLabeler(median_filter_size=filt).label({"state": state})
        diff = np.mean(base != lab)
        print(f"  filter={filt} diff={diff:.3f}")

# ---------------------------------------------------------------------------
# TEST 3: Router diagnostics - NMI vs task difficulty (CPU, reads training curves)
# ---------------------------------------------------------------------------
def test_router_nmi_vs_task():
    """
    Hypothesis D20: Can/Square NMI low (0.20/0.22 vs Lift 0.41) indicates random balanced routing.
    """
    print("\n=== TEST 3: Router NMI per task (stage2) ===")
    for task in TASKS:
        # find stage2 training_curves for seed42
        candidates = list(PHASEFORGE_STAGE2.rglob("metrics/training_curves.jsonl"))
        cand = [p for p in candidates if task in p.parent.parent.name and "seed42" in str(p)]
        if not cand:
            print(f" {task} no curve")
            continue
        p = sorted(cand)[0]
        lines = p.read_text(encoding='utf-8').splitlines()
        last = json.loads(lines[-1])
        nmi = last.get("val/phase_expert_nmi", None)
        entropy = last.get("val/routing_entropy", None)
        balance = last.get("val/topk_balance_score", None)
        print(f" {task:8} NMI={nmi:.3f} entropy={entropy:.3f} balance={balance:.3f}")
        # Check for pseudo-balancing: balance high but NMI low
        if balance and nmi:
            if balance > 0.98 and nmi < 0.25:
                print(f"  WARNING {task} pseudo-balanced (balance {balance:.2f} but NMI {nmi:.2f}) -> random routing")

# ---------------------------------------------------------------------------
# TEST 4: Stage1 phase head overfitting (CPU, reads curves)
# ---------------------------------------------------------------------------
def test_stage1_phase_overfitting():
    """
    Hypothesis C14/C15: phase head overfits late (val loss 2.7, acc drops 0.60->0.54 for Can).
    """
    print("\n=== TEST 4: Stage1 overfitting ===")
    for task in TASKS:
        candidates = list(PHASEFORGE_STAGE1.rglob("metrics/training_curves.jsonl"))
        cand = [p for p in candidates if task in p.parent.parent.name and "seed42" in str(p)]
        if not cand:
            continue
        p = sorted(cand)[0]
        lines = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
        first = lines[0]
        mid = lines[49]
        last = lines[-1]
        print(f" {task:8} epoch1 val_acc {first.get('val/phase_acc',0):.3f} val_loss_phase {first.get('val/loss_phase',0):.3f} -> epoch50 acc {mid.get('val/phase_acc',0):.3f} loss_phase {mid.get('val/loss_phase',0):.3f} -> epoch100 acc {last.get('val/phase_acc',0):.3f} loss_phase {last.get('val/loss_phase',0):.3f}")
        # Overfit signal: train acc 0.91 vs val 0.54 gap 0.37
        train_acc = last.get("train/phase_acc",0)
        val_acc = last.get("val/phase_acc",0)
        gap = train_acc - val_acc if train_acc and val_acc else 0
        if gap > 0.3:
            print(f"  WARNING {task} large train-val gap {gap:.2f} indicates overfit (lambda_phase constant 1.0)")

# ---------------------------------------------------------------------------
# TEST 5: Model forward shape on CPU (no GPU) - sanity for MoE
# ---------------------------------------------------------------------------
def test_model_forward_cpu():
    """
    Hypothesis B7/B8: MoE forward shape & router 2 experts for Square may be wrong.
    We instantiate PhaseBootstrappedMoE on CPU and forward synthetic batch.
    """
    print("\n=== TEST 5: MoE forward CPU sanity ===")
    from phaseforge.models.phase_moe import PhaseBootstrappedMoE
    from phaseforge.models.components.encoder import StateEncoder
    from phaseforge.models.components.action_head import ActionHead
    from phaseforge.models.components.phase_head import PhaseClassificationHead
    from phaseforge.models.components.router import TopKRouter
    from phaseforge.models.components.expert import ExpertMLP

    for task, sd in [("Lift",19), ("Can",23), ("Square",23)]:
        encoder = StateEncoder(input_dim=sd, hidden_dims=[256,256,256], latent_dim=128, activation="gelu", dropout=0.1)
        action_head = ActionHead(input_dim=128, output_dim=7, head_type="deterministic", hidden_dim=256)
        phase_head = PhaseClassificationHead(latent_dim=128, num_phases=6)
        router = TopKRouter(latent_dim=128, num_experts=6, top_k=2, noise_std=0.1, balance_coeff=0.01, normalize_input=True)
        expert = ExpertMLP(input_dim=128, hidden_dims=[256], output_dim=7, activation="gelu")
        model = PhaseBootstrappedMoE(encoder=encoder, action_head=action_head, phase_head=phase_head, router=router, expert=expert)
        model.eval()
        # Stage1 forward
        batch = {"state": torch.randn(4, sd), "action": torch.randn(4,7), "phase": torch.randint(0,6,(4,))}
        out = model(batch)
        assert out.action_pred.shape == (4,7), f"Stage1 action shape wrong {out.action_pred.shape}"
        assert out.phase_logits.shape == (4,6)
        print(f" {task} stage1 OK action {out.action_pred.shape} phase {out.phase_logits.shape}")
        # Bootstrap to stage2 on CPU with tiny dataloader
        # Create dummy dataloader with 8 samples
        dummy_batch = {"state": torch.randn(8, sd), "phase": torch.randint(0,6,(8,)), "trajectory_id": torch.arange(8), "trajectory_position": torch.arange(8)}
        from torch.utils.data import DataLoader, TensorDataset

        class DummyDS(torch.utils.data.Dataset):
            def __len__(self): return 12
            def __getitem__(self, idx):
                # ensure every phase 0..5 appears at least twice in 12 samples
                return {"state": torch.randn(sd), "phase": torch.tensor(idx % 6, dtype=torch.long), "trajectory_id": torch.tensor(idx%2), "trajectory_position": torch.tensor(idx)}
        dl = DataLoader(DummyDS(), batch_size=4)
        try:
            model.bootstrap_moe(dl, device="cpu", training_seed=42)
            print(f"  bootstrap to stage2 OK, stage={model.stage}")
            model.stage = 2
            # Stage2 forward
            batch2 = {"state": torch.randn(4, sd), "action": torch.randn(4,7), "phase": torch.randint(0,6,(4,)), "trajectory_id": torch.randint(0,2,(4,)), "trajectory_position": torch.randint(0,5,(4,))}
            out2 = model(batch2)
            assert out2.action_pred.shape == (4,7)
            assert out2.gate_logits.shape == (4,6)
            assert out2.expert_indices.shape == (4,2)
            print(f"  stage2 OK gate {out2.gate_logits.shape} indices {out2.expert_indices.shape} routed ok")
        except Exception as e:
            print(f"  bootstrap failed {e}")
            raise

# ---------------------------------------------------------------------------
# TEST 6: Trajectory length per task (CPU, from raw HDF5)
# ---------------------------------------------------------------------------
def test_trajectory_length():
    """
    Hypothesis: Can/Square longer/shorter than Lift -> effective steps 7500 vs 2700 imbalance in stage1.
    """
    print("\n=== TEST 6: Trajectory length per task ===")
    for task in TASKS:
        hdf5_path = TASK_TO_HDF5[task]
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            lens = []
            for demo_key in [k for k in data.keys() if k.startswith("demo_")][:20]:
                actions = data[demo_key]["actions"]
                lens.append(actions.shape[0])
            print(f" {task:8} mean_len {np.mean(lens):.0f} std {np.std(lens):.0f} min {min(lens)} max {max(lens)} total demos {len([k for k in data.keys() if k.startswith('demo_')])}")

# ---------------------------------------------------------------------------
# TEST 7: Per-phase rollout success vs phase frequency correlation
# ---------------------------------------------------------------------------
def test_per_phase_failure_correlation():
    """
    Hypothesis D21: phase5 retract failure correlates with rarity.
    """
    print("\n=== TEST 7: Per-phase success vs expected phase frequency ===")
    # Eval per-phase from earlier table: Can phase5 0.20 vs 0.28 others, Square phase5 0.033
    for task in TASKS:
        for p in sorted((EVAL_BASE / "phaseforge").rglob("eval_results.json")):
            j = json.loads(p.read_text(encoding='utf-8'))
            if j.get("eval/rollout/task") != task:
                continue
            if "seed42" not in str(p):
                continue
            per_phase = j.get("eval/rollout/per_phase_sr", {})
            print(f" {task} per_phase {per_phase}")
            # Check if phase 5 is bottleneck
            if per_phase.get("5") is not None and per_phase.get("0") is not None:
                drop = per_phase["0"] - per_phase["5"]
                if drop > 0.05:
                    print(f"  WARNING {task} phase5 drop {drop:.2f} suggests retract failure")

# ---------------------------------------------------------------------------
# TEST 8: Normalization sanity (CPU, from norm_stats.pt if exists via raw compute)
# ---------------------------------------------------------------------------
def test_normalization_sanity():
    """
    Hypothesis: object dim scale outlier for Can/Square causes zscore bias.
    We compute quick zscore from raw obs for one demo per task.
    """
    print("\n=== TEST 8: Raw obs scale per key ===")
    for task in TASKS:
        hdf5_path = TASK_TO_HDF5[task]
        with h5py.File(hdf5_path, "r") as f:
            demo_key = [k for k in f["data"].keys() if k.startswith("demo_")][0]
            obs = f["data"][demo_key]["obs"]
            for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]:
                arr = np.asarray(obs[key][:], dtype=np.float32)
                print(f" {task:8} {key:20} mean {arr.mean():.4f} std {arr.std():.4f} shape {arr.shape[1]}")
            print("---")

# ---------------------------------------------------------------------------
# Main runner for direct py execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_phase_distribution_per_task()
    test_phase_labeler_sensitivity()
    test_router_nmi_vs_task()
    test_stage1_phase_overfitting()
    test_model_forward_cpu()
    test_trajectory_length()
    test_per_phase_failure_correlation()
    test_normalization_sanity()
    print("\nAll CPU smoke tests done.")
