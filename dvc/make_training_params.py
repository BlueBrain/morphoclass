"""Generate the "training" section in params.yaml."""
from __future__ import annotations

import argparse
import itertools
import pathlib

import yaml


def main():
    """Run the main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate the "training" section in params.yaml'
    )
    parser.add_argument("--with-oversampling", action="store_true")
    parser.add_argument("--with-loo", action="store_true")
    parser.add_argument("--small-subset", action="store_true")
    args = parser.parse_args()

    config_dir = pathlib.Path("training/configs")
    dss = config_dir.glob("dataset*.yaml")
    models = config_dir.glob("model*.yaml")
    splitters = config_dir.glob("splitter*.yaml")
    small_subset = {("pc", "L2"), ("in", "L6")}
    ds_names = {"pc": "pyramidal-cells", "in": "interneurons"}

    with open("params.yaml") as fh:
        params = yaml.safe_load(fh)
    params["training"] = {}
    for ds, model, splitter in itertools.product(dss, models, splitters):
        ds_parts = ds.stem.split("-")
        if len(ds_parts) == 4:
            oversample = True
            ds_parts.pop()
        else:
            oversample = False
        _, ds_kind, layer = ds_parts
        _, model_kind, features = model.stem.split("-")
        _, splitter_kind = splitter.stem.split("-")

        if oversample and not args.with_oversampling:
            continue
        if splitter_kind == "LOO" and not args.with_loo:
            continue
        if args.small_subset and (ds_kind, layer) not in small_subset:
            continue

        id_parts = [ds_kind, features, layer, splitter_kind, model_kind]
        if oversample:
            id_parts.append("oversample")
        params["training"]["-".join(id_parts)] = {
            "dataset-dir": f"data/final/{ds_names[ds_kind]}/{layer}",
            "dataset-config": str(ds),
            "model-config": str(model),
            "splitter-config": str(splitter),
        }

    with open("params.yaml", "w") as fh:
        yaml.dump(params, fh)


if __name__ == "__main__":
    main()
