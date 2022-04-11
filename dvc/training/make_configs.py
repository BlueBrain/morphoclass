"""Generate training config files."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from dataclasses import field

import yaml

OUT_DIR = pathlib.Path(__file__).parent / "configs"


@dataclass
class Dataset:
    """Dataset config parameters."""

    id_: str
    name: str
    layers: list[str]


@dataclass
class Model:
    """Model config parameters."""

    id_: str
    name: str
    params: dict = field(default_factory=dict)
    n_epochs: int | None = None
    batch_size: int | None = None
    optimizer_class: str | None = None
    optimizer_params: dict = field(default_factory=dict)
    feature_extractors: list[str] = field(default_factory=list)


@dataclass
class Splitter:
    """Splitter config parameters."""

    id_: str
    name: str
    params: dict = field(default_factory=dict)


def make_dataset_configs():
    """Create and write dataset config files."""
    pc_layers = ["L2", "L3", "L4", "L5", "L6"]
    in_layers = ["L1", "L23", "L4", "L5", "L6"]
    datasets = [
        Dataset("in", "interneurons", in_layers),
        Dataset("pc", "pyramidal-cells", pc_layers),
    ]

    for ds in datasets:
        for layer in ds.layers:
            for oversample in [True, False]:
                config = {
                    "id": ds.id_,
                    "dataset_name": ds.name,
                    "input_csv": f"data/final/{ds.name}/{layer}/dataset.csv",
                    "layer": layer,
                    "oversampling": oversample,
                }
                file_name = f"dataset-{ds.id_}-{layer}"
                if oversample:
                    file_name += "-oversample"
                with open(OUT_DIR / f"{file_name}.yaml", "w") as fh:
                    yaml.dump(config, fh)


def make_model_configs():
    """Create and write model config files."""
    models = [
        Model(
            "decisiontree",
            "sklearn.tree.DecisionTreeClassifier",
            {"random_state": 113587},
            feature_extractors=["tmd", "deepwalk"],
        ),
        Model(
            "xgb",
            "xgboost.XGBClassifier",
            {
                "random_state": 113587,
                "objective": "multi:softmax",
                "eval_metric": "merror",
            },
            feature_extractors=["tmd", "deepwalk"],
        ),
        Model(
            "cnn",
            "morphoclass.models.CNNet",
            {"image_size": 100},
            100,
            2,
            "torch.optim.Adam",
            {"lr": 5e-3},
            ["tmd", "deepwalk"],
        ),
        Model(
            "perslay",
            "morphoclass.models.CorianderNet",
            {"n_features": 64},
            100,
            2,
            "torch.optim.Adam",
            {"lr": 5e-3, "weight_decay": 5e-4},
            ["tmd", "deepwalk"],
        ),
        Model(
            "gnn",
            "morphoclass.models.ManNet",
            {
                "n_features": 1,
                "pool_name": "avg",
                "lambda_max": 3.0,
                "normalization": "sym",
                "flow": "target_to_source",
                "edge_weight_idx": None,
            },
            150,
            2,
            "torch.optim.Adam",
            {"lr": 5e-3},
            ["morphology"],
        ),
    ]
    for model in models:
        for feature_extractor in model.feature_extractors:
            config = {
                "id": model.id_,
                "model_class": model.name,
                "model_params": model.params,
                "n_epochs": model.n_epochs,
                "batch_size": model.batch_size,
                "optimizer_class": model.optimizer_class,
                "optimizer_params": model.optimizer_params,
                "feature_extractor_name": feature_extractor,
            }
            with open(
                OUT_DIR / f"model-{model.id_}-{feature_extractor}.yaml", "w"
            ) as fh:
                yaml.dump(config, fh)


def make_splitter_configs():
    """Create and write splitter config files."""
    splitters = [
        Splitter(
            "stratifKFold",
            "sklearn.model_selection.StratifiedKFold",
            {"n_splits": 3},
        ),
        Splitter("LOO", "sklearn.model_selection.LeaveOneOut"),
    ]
    for splitter in splitters:
        config = {
            "id": splitter.id_,
            "splitter_class": splitter.name,
            "splitter_params": splitter.params,
        }
        with open(OUT_DIR / f"splitter-{splitter.id_}.yaml", "w") as fh:
            yaml.dump(config, fh)


def main():
    """Run the main program."""
    make_dataset_configs()
    make_model_configs()
    make_splitter_configs()


if __name__ == "__main__":
    main()
