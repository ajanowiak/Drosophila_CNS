# human_prepare_data/download.py

"""
Download one input file for the human prototype.

Used for the GSE244618 `.snap.gz` (with `--gunzip`, which decompresses to the
`.snap` the pipeline reads and drops the compressed copy), the Rahman loop BEDs
(kept gzipped, as `build_loop_universe` reads `.bed.gz` directly), the hg38 FASTA
(with `--gunzip --faidx`, which also writes the `.fai` index chromVAR's FaFile
needs), and the CATlas annotation metatable.

Inputs:
  - a download URL (config `snap_url` / `loops_base_url` / `genome_url` /
    `annotation_url`)
Outputs:
  - the file at --output (plus `<output>.fai` when --faidx is given)
"""

import argparse
import gzip
import logging
import shutil
import urllib.request
from pathlib import Path

from core.log import configure_logging

logger = logging.getLogger(__name__)

_CHUNK = 1 << 22


def download(url: str, dest: Path, timeout: int = 600) -> int:
    """Stream a URL to disk; return the number of bytes written."""
    with urllib.request.urlopen(url, timeout=timeout) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out, length=_CHUNK)
    return dest.stat().st_size


def gunzip(src: Path, dest: Path) -> int:
    """Decompress a .gz to dest; return the uncompressed size in bytes."""
    with gzip.open(src, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=_CHUNK)
    return dest.stat().st_size


def faidx(fasta: Path) -> None:
    """Write `<fasta>.fai` next to the FASTA (what chromVAR's FaFile expects)."""
    import pysam  # only needed for genome downloads; keeps the common path import-light
    pysam.faidx(str(fasta))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one human-prototype input file.")
    parser.add_argument("--url", required=True, help="Download URL.")
    parser.add_argument("--output", type=Path, required=True, help="Destination path.")
    parser.add_argument("--gunzip", action="store_true",
                        help="Download a .gz and decompress it to --output (drops the .gz).")
    parser.add_argument("--keep_gz", action="store_true", help="With --gunzip, keep the .gz too.")
    parser.add_argument("--faidx", action="store_true",
                        help="After writing a FASTA at --output, index it to <output>.fai (for chromVAR).")
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/download.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.gunzip:
        logger.info(f"Downloading {args.url}")
        n = download(args.url, args.output)
        logger.info(f"Wrote {args.output} ({n / 1e9:.2f} GB)")
    else:
        gz_path = Path(str(args.output) + ".gz")
        logger.info(f"Downloading {args.url}")
        gz_bytes = download(args.url, gz_path)
        logger.info(f"Downloaded {gz_bytes / 1e9:.2f} GB -> {gz_path}")

        logger.info(f"Decompressing -> {args.output}")
        out_bytes = gunzip(gz_path, args.output)
        logger.info(f"Decompressed to {out_bytes / 1e9:.2f} GB")

        if not args.keep_gz:
            gz_path.unlink(missing_ok=True)
            logger.info(f"Removed {gz_path}")

        logger.info(f"Wrote {args.output}")

    if args.faidx:
        logger.info(f"Indexing {args.output} -> {args.output}.fai")
        faidx(args.output)
        logger.info(f"Wrote {args.output}.fai")


if __name__ == "__main__":
    main()
