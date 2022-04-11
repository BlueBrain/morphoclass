"""Extract TMD neurites."""
from __future__ import annotations

import pathlib

from morphio import SectionType

from morphoclass.transforms.helper import raise_no_attribute
from morphoclass.utils import morphio_root_section_to_tmd_tree
from morphoclass.utils import print_warning


class ExtractTMDNeurites:
    """Extract neurite trees in TMD Tree format.

    Parameters
    ----------
    neurite_type : str
        Type of neurite to extract (apical, axon, basal, all)
    from_tmd_neuron : bool
        If set to true the neurite will be
        extracted from the TMD neuron in the `tmd_neuron` field of
        `data`. Otherwise, the MorphIO neuron object in the `morphology`
        field of `data` will be converted to a TMD neuron first, from
        which the neurite will be extracted.
    """

    # TODO: deprecate "neurites" in favour of "all".
    morphio_types = {
        "apical": SectionType.apical_dendrite,
        "axon": SectionType.axon,
        "basal": SectionType.basal_dendrite,
        "neurites": SectionType.all,
        "all": SectionType.all,
    }
    tmd_types = {
        "apical": "apical",
        "axon": "axon",
        "basal": "basal",
        "neurites": "neurites",
        "all": "neurites",
    }

    def __init__(self, neurite_type, from_tmd_neuron=False):
        if neurite_type not in self.morphio_types:
            raise ValueError(
                f"Unknown neurite type: {neurite_type!r}. "
                f"Possible values are {set(self.morphio_types)}"
            )
        self.neurite_type = neurite_type
        self.from_tmd_neuron = from_tmd_neuron

    def __call__(self, data):
        """Callable for TMD neurite extraction in data.

        Parameters
        ----------
        data : torch_geometric.data.data.Data
            Data instance.

        Returns
        -------
        data : torch_geometric.data.data.Data
            Processed data instance.
        """
        # Extract the neurite
        if self.from_tmd_neuron:
            if not hasattr(data, "tmd_neuron"):
                raise_no_attribute("tmd_neuron")

            data.tmd_neurites = []
            neurite_type = self.tmd_types[self.neurite_type]
            for tree in getattr(data.tmd_neuron, neurite_type):
                data.tmd_neurites.append(tree.copy_tree())
        else:
            if not hasattr(data, "morphology"):
                raise_no_attribute("morphology")

            # Find all root sections of the given type
            neurite_type = self.morphio_types[self.neurite_type]
            if neurite_type == SectionType.all:
                root_sections = data.morphology.root_sections
            else:
                root_sections = []
                for sec in data.morphology.root_sections:
                    if sec.type == neurite_type:
                        root_sections.append(sec)

            # Convert root sections to TMD trees
            data.tmd_neurites = []
            for sec in root_sections:
                tree = morphio_root_section_to_tmd_tree(sec, remove_duplicates=True)
                data.tmd_neurites.append(tree)

        # Issue a warning if neuron has no neurites of this type
        if len(data.tmd_neurites) == 0:
            if hasattr(data, "path"):
                neuron_name = pathlib.Path(data.path).stem + " "
            else:
                neuron_name = ""
            print_warning(f"Neuron {neuron_name}has no {self.neurite_type}")

        return data

    def __repr__(self):
        """Representation of the ExtractTMDNeurites class."""
        return (
            f"{self.__class__.__name__}(neurite_type={self.neurite_type},"
            f"from_tmd_neuron={self.from_tmd_neuron})"
        )
