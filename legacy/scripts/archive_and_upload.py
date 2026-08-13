"""Script to archive and upload the processed PhaseForge cache to Hugging Face."""
import argparse
import json
import re
import tarfile
from pathlib import Path

from huggingface_hub import HfApi


def make_tarball(source_dir: Path, output_filename: Path) -> None:
    print(f"Archiving {source_dir.name} -> {output_filename.name}")
    with tarfile.open(output_filename, "w:gz") as tar:
        # arcname=source_dir.name ensures it extracts as the hash folder directly
        tar.add(source_dir, arcname=source_dir.name)

def archive_and_upload(data_dir: str, repo_id: str, token: str, keep_archives: bool = False):
    cache_dir = Path(data_dir) / "processed" / "cache"
    if not cache_dir.exists():
        print(f"Error: Cache directory {cache_dir} does not exist.")
        return

    # These two small files are required on a machine that restores the
    # processed cache without the raw HDF5 files. Fail before uploading any
    # archive so the repository cannot contain a half-portable bundle.
    metadata_files = {
        "MANIFEST.json": Path(data_dir) / "raw" / "libero" / "MANIFEST.json",
        "object_index.json": Path(data_dir) / "raw" / "libero" / "object_index.json",
    }
    missing_metadata = [
        str(path) for path in metadata_files.values() if not path.is_file()
    ]
    if missing_metadata:
        print(
            "Error: required portability metadata is missing:\n  - "
            + "\n  - ".join(missing_metadata)
        )
        return

    # Find complete config-hash directories only. Do not upload temporary
    # directories or arbitrary files accidentally placed in the cache root.
    hash_dirs = []
    for candidate in sorted(cache_dir.iterdir()):
        if not candidate.is_dir() or not re.fullmatch(
            r"[0-9a-f]{16}", candidate.name
        ):
            continue
        manifest = candidate / "manifest.json"
        try:
            complete = json.loads(manifest.read_text(encoding="utf-8")).get(
                "complete", False
            )
        except (OSError, json.JSONDecodeError):
            complete = False
        if complete:
            hash_dirs.append(candidate)
    
    if not hash_dirs:
        print(f"No valid config hashes found in {cache_dir}.")
        return

    print(f"Found {len(hash_dirs)} cache directories to bundle and upload.")
    
    # Initialize API with write token
    api = HfApi(token=token)
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        print(f"Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"Failed to access repository: {e}")
        return

    for hash_dir in hash_dirs:
        tarball_path = cache_dir / f"{hash_dir.name}.tar.gz"
        
        # 1. Archive
        make_tarball(hash_dir, tarball_path)
        
        # 2. Upload
        print(f"Uploading {tarball_path.name}...")
        try:
            api.upload_file(
                path_or_fileobj=str(tarball_path),
                path_in_repo=f"data/{tarball_path.name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload packed cache {hash_dir.name}"
            )
            print(f"Uploaded {tarball_path.name} successfully. \u2705")
        except Exception as e:
            print(f"Failed to upload {tarball_path.name}: {e}")
            continue
        
        # 3. Cleanup
        if not keep_archives:
            print(f"Removing local archive {tarball_path.name} to save space...")
            tarball_path.unlink()

    # The cache key (and the eval env) depends on the dataset MANIFEST and
    # the object index; ship them alongside the archives so the same key
    # recomputes on every machine.
    extra_files = [
        (metadata_files["MANIFEST.json"], "data/MANIFEST.json"),
        (metadata_files["object_index.json"], "data/object_index.json"),
    ]
    for src, in_repo in extra_files:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {src.name}"
        )
        print(f"Uploaded {src.name} successfully. \u2705")

    print(f"\nAll operations complete! View dataset at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive and upload cache to HuggingFace")
    parser.add_argument("--repo-id", type=str, required=True, 
                        help="HuggingFace repo ID (e.g. 'username/phaseforge-cache')")
    parser.add_argument("--token", type=str, required=True, 
                        help="HuggingFace API token with WRITE access")
    parser.add_argument("--data-dir", type=str, default="data", 
                        help="Local path to the 'data' folder (default: 'data')")
    parser.add_argument("--keep-archives", action="store_true", 
                        help="Keep the generated .tar.gz files locally instead of deleting them")
    
    args = parser.parse_args()
    archive_and_upload(args.data_dir, args.repo_id, args.token, args.keep_archives)
