"""Read-only LIBERO HDF5 schema inspector.

Usage:
    python scripts/inspect_libero_hdf5.py <path_to_hdf5_file>

Prints the full group/dataset tree with shapes, dtypes,
and attribute values. No modifications to the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def print_hdf5_tree(path: Path) -> None:
    """Walk and print every group, dataset, and attribute in an HDF5 file."""
    print(f"{'='*60}")
    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / 1e6:.1f} MB")
    print(f"{'='*60}")

    with h5py.File(path, "r") as f:

        def visitor(name: str, obj):
            indent = "  " * (name.count("/") + 1)
            if isinstance(obj, h5py.Group):
                print(f"\n{indent}[GROUP] {name}")
                for attr_key, attr_val in obj.attrs.items():
                    val_str = str(attr_val)
                    if len(val_str) > 120:
                        val_str = val_str[:117] + "..."
                    print(f"{indent}  attr: {attr_key} = {val_str}")
            elif isinstance(obj, h5py.Dataset):
                dtype_str = str(obj.dtype)
                shape_str = str(obj.shape)
                # Show first few values for small datasets
                preview = ""
                if obj.size > 0 and obj.size <= 20:
                    preview = f"  preview={obj[:]}"
                elif obj.size > 0 and obj.ndim == 1:
                    preview = f"  preview={obj[:6]}"
                elif obj.size > 0 and obj.ndim == 2 and obj.shape[0] > 0:
                    preview = f"  first_row={obj[0]}"
                print(
                    f"{indent}[DATA] {name}  shape={shape_str}  "
                    f"dtype={dtype_str}{preview}"
                )

        f.visititems(visitor)

    # Also print the top-level group attrs separately
    print(f"\n{'='*60}")
    print("Top-level 'data' group analysis:")
    print(f"{'='*60}")
    data = f.get("data")
    if data is None:
        print("  NO 'data' group found!")
        return

    print(f"  Number of demo groups: {len(data)}")
    print(f"  Top-level attrs:")
    for k, v in data.attrs.items():
        val_str = str(v)
        if len(val_str) > 120:
            val_str = val_str[:117] + "..."
        print(f"    {k} = {val_str}")

    # Analyze first demo in detail
    first_demo_key = sorted(data.keys())[0]
    first_demo = data[first_demo_key]
    print(f"\n  First demo: {first_demo_key}")
    print(f"  Demo attrs:")
    for k, v in first_demo.attrs.items():
        val_str = str(v)
        if len(val_str) > 120:
            val_str = val_str[:117] + "..."
        print(f"    {k} = {val_str}")

    # Check what's in /data/demo_X/obs
    obs = first_demo.get("obs")
    if obs is not None:
        print(f"\n  Keys in obs/ subgroup:")
        for key in obs.keys():
            ds = obs[key]
            print(f"    {key}: shape={ds.shape}, dtype={ds.dtype}")

    # Check root-level datasets in demo
    print(f"\n  Root-level datasets in demo:")
    for key in first_demo.keys():
        if key == "obs":
            continue
        ds = first_demo[key]
        if isinstance(first_demo[key], h5py.Dataset):
            print(f"    {key}: shape={ds.shape}, dtype={ds.dtype}")
            if ds.ndim == 2 and ds.shape[0] > 0:
                print(f"      first row: {ds[0]}")

    # Check if robot_states exists and show its first row
    if "robot_states" in first_demo:
        rs = first_demo["robot_states"]
        print(f"\n  robot_states: shape={rs.shape}, dtype={rs.dtype}")
        print(f"    first row: {rs[0]}")
        print(f"    column breakdown hypothesis:")
        print(f"      [0:2] = gripper_qpos: {rs[0, :2]}")
        print(f"      [2:5] = eef_pos:      {rs[0, 2:5]}")
        print(f"      [5:9] = eef_quat:     {rs[0, 5:9]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_libero_hdf5.py <path_to_hdf5>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    print_hdf5_tree(path)


if __name__ == "__main__":
    main()
