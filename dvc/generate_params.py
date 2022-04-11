"""Generate DVC params.yaml file."""
from __future__ import annotations

import glob
import logging
import os
import sys
from collections import namedtuple
from itertools import product

import yaml

logger = logging.getLogger(__name__)


def main():
    """Run main."""
    # DATASETS
    dataset_row = namedtuple("dataset_row", ("id", "name", "layers"))
    if not debug:
        datasets_main = [
            dataset_row("in", "interneurons", ["L1", "L23", "L4", "L5", "L6"]),  # ALL
            dataset_row("pc", "pyramidal_cells", ["L2", "L3", "L4", "L5", "L6"]),
        ]
    else:
        datasets_main = [
            dataset_row("in", "interneurons", ["L6"]),
            dataset_row("pc", "pyramidal_cells", ["L2"]),
        ]

    params_datasets = {}
    for dataset in datasets_main:
        params_datasets[dataset.id] = make_datasets_item(dataset.id, dataset.name)

    # FEATURE EXTRACTORS
    feature_row = namedtuple("feature_row", ("id", "name"))
    feature_extractors = [
        feature_row("morph", "morphology"),
        feature_row("tmd", "tmd"),
        feature_row("deepwalk", "deepwalk"),
    ]
    params_feature_extractors = {}
    for dataset in datasets_main:
        for feat_extr in feature_extractors:
            id_ = f"{dataset.id}_{feat_extr.id}"
            item = make_feature_extractors_item(id_, dataset.name, feat_extr.name)
            params_feature_extractors[id_] = item

    # TRAIN AND EVAL
    model_row = namedtuple(
        "model_row",
        (
            "id",
            "name",
            "params",
            "n_epochs",
            "batch_size",
            "optimizer_class",
            "optimizer_params",
            "feature_extractors",
        ),
    )
    models = [
        # model_row(
        #     "adaboost",
        #     "sklearn.ensemble._weight_boosting.AdaBoostClassifier",
        #     "random_state=113587",
        #     "",
        #     "",
        #     "",
        #     "",
        #     ["tmd", "deepwalk"],
        # ),
        model_row(
            "decisiontree",
            "sklearn.tree.DecisionTreeClassifier",
            "random_state=113587",
            "",
            "",
            "",
            "",
            ["tmd", "deepwalk"],
        ),
        model_row(
            "xgb",
            "xgboost.XGBClassifier",
            "random_state=113587 objective=multi:softmax eval_metric=merror",
            "",
            "",
            "",
            "",
            ["tmd", "deepwalk"],
        ),
        model_row(
            "cnn",
            "morphoclass.models.CNNet",
            "image_size=100",
            100,
            2,
            "torch.optim.Adam",
            "lr=5e-3",
            ["tmd", "deepwalk"],
        ),
        model_row(
            "perslay",
            "morphoclass.models.CorianderNet",
            "n_features=64",
            100,
            2,
            "torch.optim.Adam",
            "lr=5e-3 weight_decay=5e-4",
            ["tmd", "deepwalk"],
        ),
        model_row(
            "gnn",
            "morphoclass.models.ManNet",
            "n_features=1 pool_name=avg lambda_max=3.0 normalization=sym "
            "flow=target_to_source edge_weight_idx=None",
            150,
            2,
            "torch.optim.Adam",
            "lr=5e-3",
            ["morphology"],
        ),
    ]

    splitter_row = namedtuple("splitter_row", ("id", "name", "params"))
    splitters = [
        splitter_row(
            "stratifKFold",
            "sklearn.model_selection.StratifiedKFold",
            "n_splits=3",
        ),
        # splitter_row("LOO", "sklearn.model_selection.LeaveOneOut", ""),
    ]

    # read best models
    best_models = []
    best_models_files = glob.glob("reports/results/**/*/best_models.txt")
    for best_models_file in best_models_files:
        with open(best_models_file) as f:
            best_models.extend(f.read().splitlines())

    logger.info("Best models found:")
    for checkpoint_path in best_models:
        logger.info(f"* {checkpoint_path}")

    oversamplings = [True, False]
    frozen_backbones = [True, False]

    params_train_and_eval = {}
    params_explain_models = {}
    params_transfer_learning = {}
    for dataset, splitter, model, oversampling in product(
        datasets_main,
        splitters,
        models,
        oversamplings,
    ):
        for feat_extr, layer in product(model.feature_extractors, dataset.layers):
            # Generate auxiliary strings
            oversampled_str = "_oversampled" if oversampling else ""
            id_ = (
                f"{dataset.id}_{feat_extr}_{layer}_"
                f"{splitter.id}_{model.id}{oversampled_str}"
            )
            dataset_name = f"{dataset.name}{f'/{layer}' if layer!= 'ALL' else ''}"
            checkpoint_path = f"models/{dataset_name}/{id_}_results.chk"

            # Populate train_and_eval
            train_and_eval_item = make_train_and_eval_item(
                id_,
                dataset_name,
                feat_extr,
                splitter,
                model,
                oversampling,
            )
            params_train_and_eval[id_] = train_and_eval_item

            if checkpoint_path not in best_models:
                continue

            # Populate explain_models
            explain_item = make_explain_models_item(id_, dataset_name)
            params_explain_models[id_] = explain_item

            if not model.name.startswith("morphoclass"):
                continue

            for dataset_tl, frozen_backbone in product(datasets_main, frozen_backbones):
                for layer_tl in dataset_tl.layers:
                    dataset_name_tl = make_dataset_name(dataset_tl, layer_tl)
                    if dataset_name == dataset_name_tl:
                        continue

                    frozen_str = "_frozen" if frozen_backbone else ""
                    dvc_item_tl = f"{id_}_TL_{dataset_tl.id}_{layer_tl}{frozen_str}"
                    dvc_item_pretrained = (
                        f"{dataset_tl.id}_{feat_extr}_{layer_tl}_"
                        f"{splitter.id}_{model.id}{oversampled_str}"
                    )
                    params_train_and_eval[dvc_item_tl] = make_train_and_eval_tl_item(
                        dvc_item_tl,
                        dvc_item_pretrained,
                        dataset_name,
                        dataset_name_tl,
                        model,
                        splitter,
                        feat_extr,
                        oversampling,
                        frozen_backbone,
                    )

            # Populate transfer_learning
            id_pre = f"*_{feat_extr}_*_{splitter.id}_{model.id}{oversampled_str}"
            tl_item = make_transfer_learning_item(
                id_,
                id_pre,
                dataset_name,
                model,
                feat_extr,
            )
            params_transfer_learning[id_] = tl_item

    # Performance report
    performance_report_params = {}
    for dataset in datasets_main:
        for layer in dataset.layers:
            id_ = f"{dataset.id}_{layer}"
            dataset_name = make_dataset_name(dataset, layer)
            performance_report_params[id_] = make_performance_report_item(dataset_name)

    params = {
        "datasets": params_datasets,
        "feature_extractors": params_feature_extractors,
        "train_and_eval": {
            "models": params_train_and_eval,
            "shared_parameters": {"seed": 113587},
        },
        "performance-report": performance_report_params,
        "explain_models": params_explain_models,
        "transfer_learning": {
            "models": params_transfer_learning,
            "shared_parameters": {"seed": 113587},
        },
    }
    with open("params.yaml", "w") as f:
        yaml.dump(params, f)


def make_dataset_name(dataset, layer):
    """Make a dataset name string."""
    dataset_name = dataset.name
    if layer != "ALL":
        dataset_name += f"/{layer}"

    return dataset_name


def make_datasets_item(id_, dataset_name):
    """Make a config item for the datasets stage."""
    return {
        "id": id_,
        "dataset_name": dataset_name,
        "input_dataset_directory": f"data/raw/{dataset_name}",
        "intermediary_directory": f"reports/data_preparation/{dataset_name}",
        "output_dataset_directory": f"data/prepared/{dataset_name}",
        "output_wildcard": f"'data/prepared/{dataset_name}/*/*/*.h5'",
    }


def make_feature_extractors_item(id_, dataset_name, feat_extr_name):
    """Make a config item for the feature_extractors stage."""
    if feat_extr_name == "morphology":
        wildcard = ""
    else:
        wildcard = f"'data/prepared/{dataset_name}/*/*/*_{feat_extr_name}.pickle'"

    return {
        "id": id_,
        "dataset_name": dataset_name,
        "input_csv": f"data/prepared/{dataset_name}/dataset_features_labels.csv",
        "feature_extractor_name": feat_extr_name,
        "output_wildcard": wildcard,
    }


def make_transfer_learning_item(
    item_id,
    item_id_pretrained,
    dataset_name,
    model,
    feature_extractor,
):
    """Make a config item for the transfer_learning stage."""
    checkpoint_paths_pretained = f"models/*/*/{item_id_pretrained}_results.chk"
    csv_dir = f"data/prepared/{dataset_name}"
    report_dir = f"reports/transfer_learning/{dataset_name}"
    return {
        "id": item_id,
        "results_file": f"{report_dir}/transfer_learning_report.html",
        "model_class": model.name,
        "model_params": model.params,
        "n_epochs": model.n_epochs,
        "batch_size": model.batch_size,
        "optimizer_class": model.optimizer_class,
        "optimizer_params": model.optimizer_params,
        "dataset_name": dataset_name,
        "input_csv": f"{csv_dir}/dataset_features_labels.csv",
        "feature_extractor_name": feature_extractor,
        "checkpoint_paths_pretrained": checkpoint_paths_pretained,
        # "frozen_backbone": frozen_backbone,
    }


def make_explain_models_item(item_id, dataset_name):
    """Make a config item for the explain_models stage."""
    return {
        "id": item_id,
        "results_file": f"reports/xai/{dataset_name}/{item_id}/xai_report.html",
        "checkpoint_path": f"models/{dataset_name}/{item_id}_results.chk",
        "checkpoint_path_outs": f"reports/xai/{dataset_name}/{item_id}/",
    }


def make_train_and_eval_item(
    dvc_item,
    dataset_name,
    feature_extractor,
    splitter,
    model,
    oversampling,
):
    """Make a config item for the train_and_eval stage."""
    checkpoint_path = f"models/{dataset_name}/{dvc_item}_results.chk"
    return {
        "id": dvc_item,
        "model_class": model.name,
        "model_params": model.params,
        "n_epochs": model.n_epochs,
        "batch_size": model.batch_size,
        "optimizer_class": model.optimizer_class,
        "optimizer_params": model.optimizer_params,
        "splitter_class": splitter.name,
        "splitter_params": splitter.params,
        "dataset_name": dataset_name,
        "input_csv": f"data/prepared/{dataset_name}/dataset_features_labels.csv",
        "feature_extractor_name": feature_extractor,
        "images_directory": f"models/{dataset_name}/images/{dvc_item}/",
        "checkpoint_path": checkpoint_path,
        "oversampling": oversampling,
        "train_all_samples": True,
        "checkpoint_path_pretrained": "",
        "frozen_backbone": False,
        "output_wildcard": f"'reports/results/{dataset_name}/{dvc_item}/images/*'",
    }


def make_train_and_eval_tl_item(
    id_,
    id_pre,
    dataset_name,
    dataset_name_tl,
    model,
    splitter,
    feat_extr,
    oversampling,
    frozen_backbone,
):
    """Make a config item for the train_and_eval transfer learning stage."""
    checkpoint_path = f"models/{dataset_name}/{id_}_results.chk"
    checkpoint_path_pre = f"models/{dataset_name_tl}/{id_pre}_results.chk"
    return {
        "id": id_,
        "model_class": model.name,
        "model_params": model.params,
        "n_epochs": model.n_epochs,
        "batch_size": model.batch_size,
        "optimizer_class": model.optimizer_class,
        "optimizer_params": model.optimizer_params,
        "splitter_class": splitter.name,
        "splitter_params": splitter.params,
        "dataset_name": dataset_name,
        "input_csv": f"data/prepared/{dataset_name}/dataset_features_labels.csv",
        "feature_extractor_name": feat_extr,
        "images_directory": f"models/{dataset_name}/images/{id_}/",
        "checkpoint_path": checkpoint_path,
        "oversampling": oversampling,
        "train_all_samples": True,
        "checkpoint_path_pretrained": checkpoint_path_pre,
        "frozen_backbone": frozen_backbone,
        "output_wildcard": f"'reports/results/{dataset_name}/{id_}/images/*'",
    }


def make_performance_report_item(dataset_name):
    """Make a config item for the results stage."""
    return {
        "checkpoint_dir": f"models/{dataset_name}",
        "output_dir": f"reports/results/{dataset_name}",
    }


if __name__ == "__main__":
    debug = os.getenv("MORPHOCLASS_ENV") == "debug"
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    if debug:
        logger.info("generating a DEBUG version of params.yaml")
    else:
        logger.info("generating the NORMAL (full) version of params.yaml")

    sys.exit(main())
