# human_prepare_data/inspect_snap.py

"""Dump the HDF5 group/dataset structure of a SNAP file, for schema discovery."""

import argparse

import h5py
import numpy as np


def walk(group: h5py.Group, prefix: str = "") -> None:
    """Recursively print groups and datasets (with a small value sample)."""
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Group):
            print(f"[GROUP] {path}")
            walk(item, path)
        else:
            sample = np.asarray(item[:5] if item.size > 5 else item[()]).tolist() if item.size else ""
            print(f"  [DSET] {path} shape={item.shape} dtype={item.dtype} e.g. {sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the HDF5 structure of a SNAP file.")
    parser.add_argument("snap", help="Path to a .snap file.")
    args = parser.parse_args()
    with h5py.File(args.snap, "r") as f:
        print("=== TOP-LEVEL KEYS ===", list(f.keys()))
        walk(f)


if __name__ == "__main__":
    main()
