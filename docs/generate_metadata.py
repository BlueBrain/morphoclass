"""Generate metadata files for the BBP software catalog."""
from __future__ import annotations

import pathlib

import sphinx_bluebrain_theme.utils.metadata as md

contributors = [
    "Stanislav Schmidt",
    "Emilie Delattre",
    "Francesco Casalegno",
]

metadata = md.build_metadata(
    distribution_name="morphoclass",
    metadata_overrides={
        "name": "Morphology-Classification",
        "contributors": ", ".join(contributors),
    },
)
md.write_metadata(metadata=metadata, output_dir=pathlib.Path(__file__).parent)
