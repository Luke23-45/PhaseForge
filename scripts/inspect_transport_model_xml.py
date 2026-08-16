"""Inspect the exact robosuite model XML embedded in the Transport HDF5.

This is an offline provenance check for geometry-sensitive rollout debugging;
it does not instantiate MuJoCo or modify the dataset.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET

import h5py

DEFAULT_DATASET = "data/raw/robomimic/transport/low_dim_v15.hdf5"


def inspect(path: str) -> None:
    with h5py.File(path, "r") as h5:
        data = h5["data"]
        demo_name = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))[0]
        model_xml = data[demo_name].attrs["model_file"]
        env_args = data.attrs["env_args"]

    root = ET.fromstring(model_xml)
    payload = root.find(".//body[@name='payload_root']")
    if payload is None:
        raise ValueError("embedded model XML has no payload_root body")
    head = payload.find("./geom[@name='payload_head']")
    if head is None:
        raise ValueError("embedded model XML has no payload_head geom")

    pos = [float(value) for value in head.attrib["pos"].split()]
    if len(pos) != 3:
        raise ValueError(f"payload_head pos must have 3 values, got {pos}")

    args = json.loads(env_args) if isinstance(env_args, str) else env_args
    print(f"dataset: {path}")
    print(f"sample model: {demo_name}")
    print(f"env: {args['env_name']} {args['env_version']}")
    print(f"payload_head local pos: {pos}")
    print(f"payload_head local-axis offset: {pos[2]:.6f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_DATASET)
    args = parser.parse_args()
    inspect(args.path)


if __name__ == "__main__":
    main()
