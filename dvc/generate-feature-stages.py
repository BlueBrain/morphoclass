"""Generate DVC feature extraction stages."""
from __future__ import annotations

import re
from itertools import product

import yaml

neurites = ["apical", "axon", "basal", "all"]
features = [
    "graph-rd",
    "diagram-tmd-rd",
    "diagram-deepwalk",
    "image-tmd-rd",
    "image-deepwalk",
]
datasets = {
    "in-L1": "data/final/interneurons/L1/dataset.csv",
    "in-L23": "data/final/interneurons/L23/dataset.csv",
    "in-L4": "data/final/interneurons/L4/dataset.csv",
    "in-L5": "data/final/interneurons/L5/dataset.csv",
    "in-L6": "data/final/interneurons/L6/dataset.csv",
    "pc-L2": "data/final/pyramidal-cells/L2/dataset.csv",
    "pc-L3": "data/final/pyramidal-cells/L3/dataset.csv",
    "pc-L4": "data/final/pyramidal-cells/L4/dataset.csv",
    "pc-L5": "data/final/pyramidal-cells/L5/dataset.csv",
    "pc-L6": "data/final/pyramidal-cells/L6/dataset.csv",
    "lida-in-merged": "data/final/IN_data.csv",
    "lida-in-merged-bc-merged": "data/final/IN_data_BC_merged.csv",
    "lida-janelia-L5": "data/final/classes-janelia-L5.csv",
}
deps = {
    "in-L1": ["data/final/interneurons/L1"],
    "in-L23": ["data/final/interneurons/L23"],
    "in-L4": ["data/final/interneurons/L4"],
    "in-L5": ["data/final/interneurons/L5"],
    "in-L6": ["data/final/interneurons/L6"],
    "pc-L2": ["data/final/pyramidal-cells/L2"],
    "pc-L3": ["data/final/pyramidal-cells/L3"],
    "pc-L4": ["data/final/pyramidal-cells/L4"],
    "pc-L5": ["data/final/pyramidal-cells/L5"],
    "pc-L6": ["data/final/pyramidal-cells/L6"],
    "lida-in-merged": ["data/final/IN_data.csv", "data/final/interneurons"],
    "lida-in-merged-bc-merged": [
        "data/final/IN_data_BC_merged.csv",
        "data/final/interneurons",
    ],
    "lida-janelia-L5": ["data/final/classes-janelia-L5.csv", "data/final/janelia"],
}

stages = {}
for neurite, feature, (ds, csv) in product(neurites, features, datasets.items()):
    # No apicals in interneurons
    if "in-" in ds and neurite == "apical":
        continue

    # Use projection feature instead of radial distances for L2/L6 pyramidal cells
    if re.search(r"pc-L[26]|janelia-L[26]", ds):
        feature = feature.replace("-rd", "-proj")
        print(f"Using projection features {feature} for dataset {ds}")
    config = {
        "cmd": (
            f"morphoclass -v extract-features {csv} {neurite} {feature} "
            f"extract-features/{ds}/{neurite}/{feature}"
        ),
        "deps": deps[ds],
        "outs": [f"extract-features/{ds}/{neurite}/{feature}"],
    }
    stages[f"features-{ds}-{feature}-{neurite}"] = config


class NoAliasDumper(yaml.SafeDumper):
    """A dumper that doesn't create aliases in YAML files."""

    def ignore_aliases(self, data):
        """Ignore aliases."""
        return True


print(f"Creating {len(stages)} stages")
with open("dvc-feature-stages.yaml", "w") as fh:
    yaml.dump({"stages": stages}, fh, Dumper=NoAliasDumper)
