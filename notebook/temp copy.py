from pathlib import Path
import hashlib
import shutil

import h5py
from huggingface_hub import hf_hub_download, get_paths_info

REPO_ID = "amandlek/robomimic"
REPO_TYPE = "dataset"
REVISION = "2aa5bedb20cb20d24b7d857035643cb5deaadbba"

DATA_ROOT = Path("C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/data/raw/robomimic")

TASK_FILES = {
    "lift": "v1.5/lift/ph/low_dim_v15.hdf5",
    "can": "v1.5/can/ph/low_dim_v15.hdf5",
    "square": "v1.5/square/ph/low_dim_v15.hdf5",
    "tool_hang": "v1.5/tool_hang/ph/low_dim_v15.hdf5",
    "transport": "v1.5/transport/ph/low_dim_v15.hdf5",
}

KNOWN_SHA256 = {
    "lift": "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_sha256(filename: str) -> str | None:
    info = get_paths_info(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        paths=filename,
        revision=REVISION,
    )
    if not info:
        return None

    lfs = getattr(info[0], "lfs", None)
    return getattr(lfs, "sha256", None) if lfs is not None else None


for task, filename in TASK_FILES.items():
    destination_dir = DATA_ROOT / task
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "low_dim_v15.hdf5"

    expected_sha = KNOWN_SHA256.get(task) or remote_sha256(filename)

    if destination.exists():
        actual_sha = sha256_file(destination)
        if expected_sha is None:
            raise RuntimeError(
                f"No remote SHA-256 metadata available for {task}; "
                f"refusing to trust existing file {destination}."
            )
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Existing {destination} has SHA-256 {actual_sha}, "
                f"expected {expected_sha}."
            )
        print(f"{task}: existing file verified")
    else:
        cached = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            revision=REVISION,
        )
        shutil.copy2(cached, destination)

        actual_sha = sha256_file(destination)
        if expected_sha is None:
            raise RuntimeError(
                f"No remote SHA-256 metadata available for {task}; "
                f"downloaded file was not accepted."
            )
        if actual_sha != expected_sha:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded {destination} has SHA-256 {actual_sha}, "
                f"expected {expected_sha}."
            )

    with h5py.File(destination, "r") as h5:
        if "data" not in h5:
            raise RuntimeError(f"{destination} is missing the required 'data' group.")
        if "env_args" not in h5["data"].attrs:
            raise RuntimeError(f"{destination} is missing data.attrs['env_args'].")

    print(f"{task}: {destination}")
    print(f"{task}: sha256={sha256_file(destination)}")

print("All five low-dimensional robomimic datasets are verified.")