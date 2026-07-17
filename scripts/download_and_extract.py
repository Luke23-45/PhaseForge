"""Script to download and extract PhaseForge cache from Hugging Face."""
import argparse
import os
import tarfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

def extract_tarball(tarball_path: Path, extract_dir: Path) -> None:
    print(f"Extracting {tarball_path.name}...")
    with tarfile.open(tarball_path, "r:gz") as tar:
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
                local_dir_use_symlinks=False
            )
            
            downloaded_tarball = Path(local_path)
            
            extract_tarball(downloaded_tarball, cache_dir)
            
            if not keep_archives:
                print(f"Cleaning up {filename} to save space...")
                downloaded_tarball.unlink()
                
                # Cleanup empty 'data' dir if hf_hub_download created it
                if downloaded_tarball.parent.name == "data" and not list(downloaded_tarball.parent.iterdir()):
                    downloaded_tarball.parent.rmdir()
                    
            print(f"Successfully downloaded and extracted {filename} \u2705")
        except Exception as e:
            print(f"Failed on {filename}: {e}")

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
