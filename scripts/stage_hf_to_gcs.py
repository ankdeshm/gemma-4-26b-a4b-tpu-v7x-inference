#!/usr/bin/env python3
"""Zero-local-disk, idempotent, multi-threaded Hugging Face -> GCS weight streamer.

Derived from the stage_hf_to_gcs.py used in the earlier vllm-torchtpu repos, plus:
  * --exclude / --include glob filters (gpt-oss-120b ships two extra 65 GB copies of
    the weights under original/ and metal/ that no PyTorch server reads),
  * a post-copy verification pass that compares every GCS object size with the
    Hugging Face size and writes a manifest (_STAGE_MANIFEST.json) next to the weights.

Usage:
  python3 stage_hf_to_gcs.py --bucket <YOUR_GCS_BUCKET> --prefix models \
      --repos openai/gpt-oss-120b --exclude 'original/*,metal/*'
"""

import argparse
import concurrent.futures
import fnmatch
import json
import os
import time

import requests
from google.cloud import storage
from huggingface_hub import HfApi, hf_hub_url


def wanted(name: str, includes: list[str], excludes: list[str]) -> bool:
    if name.startswith(".git"):
        return False
    if includes and not any(fnmatch.fnmatch(name, p) for p in includes):
        return False
    return not any(fnmatch.fnmatch(name, p) for p in excludes)


PART_BYTES = 1536 * 1024 * 1024  # 1.5 GiB parts for large files (<=32 parts per compose call)


def _copy_range(url, headers, bucket, part_name, start, end):
    """Copies bytes [start, end] of url into GCS object part_name, retrying the whole part."""
    for attempt in range(1, 6):
        try:
            h = dict(headers)
            h["Range"] = f"bytes={start}-{end}"
            blob = bucket.blob(part_name)
            written = 0
            with requests.get(url, headers=h, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with blob.open("wb", chunk_size=32 * 1024 * 1024) as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
            if written != end - start + 1:
                raise IOError(f"short part {written} != {end - start + 1}")
            return written
        except Exception as exc:  # noqa: BLE001
            print(f"[stage-gcs] WARN part {part_name} attempt {attempt}/5: {exc}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"part failed: {part_name}")


def transfer_large(url, headers, bucket, blob_path, size, part_workers=12):
    """Parallel ranged download -> temporary part objects -> GCS compose (handles >32 parts in levels)."""
    ranges = [(i, s, min(s + PART_BYTES, size) - 1) for i, s in enumerate(range(0, size, PART_BYTES))]
    parts = [f"{blob_path}.__part{i:04d}" for i, _, _ in ranges]
    with concurrent.futures.ThreadPoolExecutor(max_workers=part_workers) as pool:
        list(pool.map(lambda r: _copy_range(url, headers, bucket, parts[r[0]], r[1], r[2]), ranges))
    level, names = 0, parts
    while len(names) > 32:
        nxt = []
        for j in range(0, len(names), 32):
            tmp = f"{blob_path}.__lvl{level}_{j // 32:04d}"
            bucket.blob(tmp).compose([bucket.blob(n) for n in names[j:j + 32]])
            nxt.append(tmp)
        for n in names:
            bucket.blob(n).delete()
        names, level = nxt, level + 1
    final = bucket.blob(blob_path)
    final.compose([bucket.blob(n) for n in names])
    for n in names:
        bucket.blob(n).delete()
    final.reload()
    return final.size


def transfer_one_file(repo_id, revision, rfilename, expected_size, bucket_name, prefix, hf_token):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_path = f"{prefix.rstrip('/')}/{repo_id}/{rfilename}"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.reload()
        if expected_size > 0 and blob.size == expected_size:
            return rfilename, "SKIPPED_EXISTS", blob.size
    url = hf_hub_url(repo_id=repo_id, filename=rfilename, revision=revision)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    if expected_size > 2 * 1024 ** 3:
        t0 = time.time()
        n = transfer_large(url, headers, bucket, blob_path, expected_size)
        dt = max(time.time() - t0, 1e-3)
        print(f"[stage-gcs] DONE(ranged) {repo_id}/{rfilename} ({n/1e9:.2f} GB in {dt:.1f}s = "
              f"{n/dt/2**20:.1f} MiB/s)", flush=True)
        return rfilename, "UPLOADED_RANGED", n
    for attempt in range(1, 5):
        t0, written = time.time(), 0
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with blob.open("wb", chunk_size=64 * 1024 * 1024) as gcs_f:
                    for chunk in resp.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            gcs_f.write(chunk)
                            written += len(chunk)
            dt = max(time.time() - t0, 1e-3)
            print(f"[stage-gcs] DONE {repo_id}/{rfilename} ({written/1e9:.2f} GB in {dt:.1f}s = "
                  f"{written/dt/2**20:.1f} MiB/s)", flush=True)
            return rfilename, "UPLOADED", written
        except Exception as exc:  # noqa: BLE001
            print(f"[stage-gcs] WARN attempt {attempt}/4 failed for {repo_id}/{rfilename}: {exc}", flush=True)
            if attempt == 4:
                raise
            time.sleep(10 * attempt)
    return rfilename, "FAILED", 0


def stage_repo(repo_id, bucket_name, prefix, workers, hf_token, includes, excludes):
    api = HfApi(token=hf_token)
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    siblings = [s for s in info.siblings if wanted(s.rfilename, includes, excludes)]
    skipped = [s.rfilename for s in info.siblings if not wanted(s.rfilename, includes, excludes)]
    total = sum((s.size or 0) for s in siblings)
    print(f"[stage-gcs] === {repo_id}@{info.sha} -> gs://{bucket_name}/{prefix}/{repo_id}/ "
          f"({len(siblings)} files, {total/1e9:.2f} GB, excluded {len(skipped)} files, workers={workers}) ===",
          flush=True)
    t_repo, moved, done = time.time(), 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(transfer_one_file, repo_id, info.sha, s.rfilename, s.size or 0, bucket_name, prefix,
                            hf_token): s.rfilename for s in siblings}
        for fut in concurrent.futures.as_completed(futs):
            _, _, nbytes = fut.result()
            done += 1
            moved += nbytes
            if done % 5 == 0 or done == len(siblings):
                print(f"[stage-gcs] Progress {repo_id}: {done}/{len(siblings)} files "
                      f"({moved/1e9:.1f}/{total/1e9:.1f} GB, {time.time()-t_repo:.0f}s)", flush=True)

    # Verification: every expected object exists with the exact Hugging Face byte size.
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    base = f"{prefix.rstrip('/')}/{repo_id}/"
    gcs_sizes = {b.name[len(base):]: b.size for b in client.list_blobs(bucket_name, prefix=base)}
    mismatches = [
        {"file": s.rfilename, "hf_size": s.size, "gcs_size": gcs_sizes.get(s.rfilename)}
        for s in siblings
        if gcs_sizes.get(s.rfilename) != s.size
    ]
    manifest = {
        "repo_id": repo_id,
        "revision": info.sha,
        "gcs_uri": f"gs://{bucket_name}/{base}",
        "files": len(siblings),
        "bytes": total,
        "excluded_patterns": excludes,
        "excluded_files": skipped,
        "verified_ok": not mismatches,
        "mismatches": mismatches,
        "completed_at_unix": time.time(),
        "per_file": {s.rfilename: s.size for s in siblings},
    }
    bucket.blob(base + "_STAGE_MANIFEST.json").upload_from_string(json.dumps(manifest, indent=1))
    status = "VERIFIED" if not mismatches else f"MISMATCH({len(mismatches)})"
    print(f"[stage-gcs] === {status} {repo_id}: {len(siblings)} files, {total/1e9:.2f} GB, "
          f"{time.time()-t_repo:.0f}s ===", flush=True)
    if mismatches:
        raise SystemExit(f"verification failed for {repo_id}: {mismatches[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", required=True, help="Comma-separated HF repo ids, staged in order")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="models")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--include", default="", help="Comma-separated globs; empty = everything")
    ap.add_argument("--exclude", default="", help="Comma-separated globs, e.g. 'original/*,metal/*'")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    inc = [p for p in args.include.split(",") if p]
    exc = [p for p in args.exclude.split(",") if p]
    for repo in [r.strip() for r in args.repos.split(",") if r.strip()]:
        stage_repo(repo, args.bucket, args.prefix, args.workers, token, inc, exc)


if __name__ == "__main__":
    main()
