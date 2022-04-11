"""Preprocess the Janelia dataset CSV file.

The purpose of this script is to transform the dataset CSV file so that it
is compatible with the MCAR curation. The MCAR curation expects a dataset
CSV file with the columns `morph_name`, `morph_path`, and `mtype`.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

logger = logging.getLogger(pathlib.Path(__file__).name)


def main() -> int:
    """Run the script."""
    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_csv_path", type=pathlib.Path)
    parser.add_argument("output_csv_path", type=pathlib.Path)
    parser.add_argument("dvc_dir", type=pathlib.Path)
    args = parser.parse_args()

    if not args.dataset_csv_path.exists():
        logger.error(f"The file {str(args.dataset_csv_path)!r} doesn't exist.")
        return 1
    if args.output_csv_path.exists():
        logger.error(f"The output file {str(args.output_csv_path)!r} already exists.")
        return 1
    if not args.dvc_dir.exists():
        logger.error(f"The directory {str(args.dvc_dir)!r} doesn't exist.")
        return 1
    if not args.dvc_dir.is_dir():
        logger.error(f"The dvc_dir argument {str(args.dvc_dir)!r} is not a directory.")
        return 1

    logger.info("Reading dataset CSV")
    df = pd.read_csv(args.dataset_csv_path)
    df = df.rename(columns={"path": "morph_path", "label": "mtype"})

    logger.info("Processing dataset CSV")
    df["morph_name"] = df["morph_path"].map(lambda p: pathlib.Path(p).stem)
    dataset_dir = args.dataset_csv_path.parent.resolve()
    dvc_dir = args.dvc_dir.resolve()
    df["morph_path"] = df["morph_path"].map(
        lambda p: (dataset_dir / p).relative_to(dvc_dir)
    )

    logger.info(f"Writing processed CSV to {args.output_csv_path.resolve().as_uri()}")
    df.to_csv(
        args.output_csv_path,
        index=False,
        columns=("morph_name", "morph_path", "mtype"),
    )

    logger.info("Done")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
