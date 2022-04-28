# Datasets

## Pyramidal Cells
See [pyramidal-cells/README.md](pyramidal-cells/README.md)

## Interneurons (INs)
See [interneurons/README.md](interneurons/README.md)

## Janelia
This is a dataset of pyramidal cells obtained from Ying via Lida.

* Original path:
  `/gpfs/bbp.cscs.ch/data/project/proj78/janelia_oriented_for_rodrigo/WholeBrain/mouselight_isocortex_ASCII_Files`
* Some files had a lowercase extension ".asc", which was changed to upper case
  ".ASC" to match the rest of the files
* Based on the notes in both Excel-files a subset of morphology files was
  selected to compose a dataset. This dataset is represented by the file
  `selected_dataset.csv`.

### Janelia inter-rater annotations `janelia/inter-rater`

All the "ground truth" labels used in this project for training and evaluation
purposes were provided by Ying Shi. But in order to estimate the inter-rater
agreement between different experts, Julie Meystre also provided her own labels
(independently and without knowing anything about Ying Shi's labels) on the
Layer 5 Janelia pyramidal cells.
* `Classification_JulieM.xlsx` - This is the original Excel spreadsheet
  provided by Julie Meystre, including her labels as well as per-sample
  comments (in free text format).
* `inter_rater-janelia-l5-ying_shi-julie_meystre.csv` - This is a CSV file
  derived from the aforementioned Excel spreadsheet. It summarizes the
  information about the labels given by the two human experts, in order to
  allow for easy evaluation of the inter-rater agreement.
