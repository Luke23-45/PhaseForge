"""Script to download and extract PhaseForge cache from Hugging Face."""
import argparse
import os
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def extract_tarball(tarball_path: Path, extract_dir: Path) -> None:
    print(f"Extracting {tarball_path.name}...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        root = extract_dir.resolve()
        for member in tar.getmembers():
            # Archives are downloaded from a remote repository. Reject
            # traversal paths and links before extraction; the cache bundle
            # should contain regular files/directories only.
            target = (extract_dir / member.name).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise RuntimeError(
                    f"Unsafe archive member {member.name!r}: it escapes "
                    f"the extraction directory {root}"
                )
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Unsafe archive member {member.name!r}: links are not "
                    "allowed in a cache bundle"
                )
        tar.extractall(path=extract_dir)

def download_and_extract(repo_id: str, data_dir: str, token: str, keep_archives: bool = False):
    cache_dir = Path(data_dir) / "processed" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    api = HfApi(token=token)
    
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Failed to access repository: {e}")
        return
        
    tarball_files = [f for f in files if f.startswith("data/") and f.endswith(".tar.gz")]
    
    if not tarball_files:
        print(f"No .tar.gz files found in the 'data/' directory of {repo_id}.")
        return
        
    print(f"Found {len(tarball_files)} archives to download.")
    
    for file_path in tarball_files:
        filename = Path(file_path).name
        print(f"\nDownloading {filename}...")
        
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                token=token,
                local_dir=str(cache_dir),
            )
            
            downloaded_tarball = Path(local_path)
            
            extract_tarball(downloaded_tarball, cache_dir)
            
            if not keep_archives:
                print(f"Cleaning up {filename} to save space...")
                downloaded_tarball.unlink()
                
                # Cleanup empty 'data' dir if hf_hub_download created it
                parent_has_children = (
                    downloaded_tarball.parent.name == "data"
                    and not list(downloaded_tarball.parent.iterdir())
                )
                if parent_has_children:
                    downloaded_tarball.parent.rmdir()
                    
            print(f"Successfully downloaded and extracted {filename} \u2705")
        except Exception as e:
            print(f"Failed on {filename}: {e}")

    # The cache key (and the eval env) needs the dataset MANIFEST and the
    # object index at their original paths; fetch them alongside the archives.
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for in_repo, target in [
        ("data/MANIFEST.json", "libero/MANIFEST.json"),
        ("data/object_index.json", "libero/object_index.json"),
    ]:
        if in_repo not in files:
            print(
                f"Warning: {in_repo} not in the repo — "
                "the cache key may not match on this machine."
            )
            continue
        local = hf_hub_download(
            repo_id=repo_id,
            filename=in_repo,
            repo_type="dataset",
            token=token,
            local_dir=str(raw_dir),
        )
        destination = raw_dir / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        Path(local).replace(destination)
        print(f"Downloaded {Path(in_repo).name} \u2705")

    stale_data_dir = raw_dir / "data"
    if stale_data_dir.exists() and not list(stale_data_dir.iterdir()):
        stale_data_dir.rmdir()

    print("\nAll downloads and extractions complete! Your cache is ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract cache from HuggingFace")
    parser.add_argument("--repo-id", type=str, required=True, 
                        help="HuggingFace repo ID (e.g. 'username/phaseforge-cache')")
    parser.add_argument("--token", type=str, required=False, default=None,
                        help="HuggingFace API token (only required if repo is private)")
    parser.add_argument("--data-dir", type=str, default="data", 
                        help="Local path to extract to (default: 'data')")
    parser.add_argument("--keep-archives", action="store_true", 
                        help="Keep the downloaded .tar.gz files locally instead of deleting them")
    
    args = parser.parse_args()
    download_and_extract(args.repo_id, args.data_dir, args.token, args.keep_archives)
