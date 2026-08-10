#!/usr/bin/env python3
"""copy_mask.py

Copy every folder under a source tree whose *name* contains "mask"
(case-insensitive substring match) into a destination tree, preserving the
folder's path relative to the source root.

For each matched folder:
  - The matched folder itself is copied in full (all files and
    subfolders inside it), recursively.
  - Its ancestor folders (from the source root down to the match) are
    recreated as empty directories only -- their other files/subfolders
    are NOT copied.
  - A folder nested inside an already-matched folder is not treated as a
    separate match; it's already included in the full copy of its parent
    match, so traversal does not descend into a matched folder looking
    for further matches.

The destination tree is not wiped before running. Existing files at the
same relative path are overwritten; anything else already in the
destination is left untouched.

Usage:
    python3 copy_mask.py [--src /videos] [--dst ./masks] [--dry-run]
"""

import argparse
import os
import shutil
import sys


def find_matches_and_copy(src_root: str, dst_root: str, dry_run: bool) -> list[str]:
    """Walk src_root, copy every "*mask*"-named folder to dst_root.

    Returns the list of matched folders (as paths relative to src_root).
    """
    copied: list[str] = []

    # Edge case: the source root itself matches -- copy the whole tree
    # in one shot and stop, there's nothing else to walk.
    if "mask" in os.path.basename(os.path.normpath(src_root)).lower():
        rel = "."
        copied.append(rel)
        print(f"{'[dry-run] ' if dry_run else ''}copy: {src_root} -> {dst_root}")
        if not dry_run:
            os.makedirs(dst_root, exist_ok=True)
            shutil.copytree(src_root, dst_root, dirs_exist_ok=True)
        return copied

    for dirpath, dirnames, _filenames in os.walk(src_root, topdown=True):
        matched_here = [d for d in dirnames if "mask" in d.lower()]

        for name in matched_here:
            src_dir = os.path.join(dirpath, name)
            rel = os.path.relpath(src_dir, src_root)
            dst_dir = os.path.join(dst_root, rel)

            copied.append(rel)
            print(f"{'[dry-run] ' if dry_run else ''}copy: {rel}")

            if not dry_run:
                # Recreate the parent chain as empty directories only.
                os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                # Copy the matched folder in full, merging/overwriting
                # into an existing destination if present.
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        # Prune: don't descend into folders we've already copied in full.
        dirnames[:] = [d for d in dirnames if d not in matched_here]

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy all '*mask*'-named folders (with their subfolders "
        "and relative parent path) from --src into --dst."
    )
    parser.add_argument(
        "--src", default="/videos", help="source root to search (default: /videos)"
    )
    parser.add_argument(
        "--dst",
        default="./masks",
        help="destination root to copy into (default: ./masks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be copied without actually copying anything",
    )
    args = parser.parse_args()

    src_root = os.path.abspath(args.src)
    dst_root = os.path.abspath(args.dst)

    if not os.path.isdir(src_root):
        print(f"Error: source directory does not exist: {src_root}", file=sys.stderr)
        return 1

    copied = find_matches_and_copy(src_root, dst_root, args.dry_run)

    verb = "would copy" if args.dry_run else "copied"
    print(f"\nDone: {verb} {len(copied)} folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
