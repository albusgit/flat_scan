"""upload.py BUNDLE_DIR USER/REPO: the stage.py bundle (code.tar, weights.tar, ...) to a private Hugging Face dataset repo,
before renting: the box then pulls it at datacentre speed (box.sh: STORE=hf:USER/REPO) instead of waiting on an upload
from home. Log in first on this PC (the token stays with you):  hf auth login
upload_large_folder sends in parallel and resumes: run it again after a break and it skips what is already up."""
import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bundle'); ap.add_argument('repo'); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.bundle) if f.endswith('.tar'))
    tot = sum(os.path.getsize(os.path.join(a.bundle, f)) for f in files)
    print(f'{len(files)} parts, {tot / 2 ** 30:.2f} GB:', ', '.join(f'{f} {os.path.getsize(os.path.join(a.bundle, f)) / 2 ** 30:.2f}' for f in files), flush=True)
    api = HfApi()
    print('as', api.whoami()['name'], flush=True)
    api.create_repo(a.repo, repo_type='dataset', private=True, exist_ok=True)
    if not api.repo_info(a.repo, repo_type='dataset').private:
        raise SystemExit(f'{a.repo} is public: the bundle goes only to a private repo')
    api.upload_large_folder(repo_id=a.repo, folder_path=a.bundle, repo_type='dataset', allow_patterns=['*.tar', 'box.sh'], num_workers=a.workers)
    have = set(api.list_repo_files(a.repo, repo_type='dataset'))
    miss = [f for f in files if f not in have]
    print('missing:', miss if miss else 'none: every part is up', flush=True)


if __name__ == '__main__':
    main()
