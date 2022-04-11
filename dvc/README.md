# DVC Pipelines Summary

The following stages are the part of DVC pipeline:
1. Data preparation
2. Feature extraction
3. Train and evaluate
4. Results
5. Explanation

Note:
- `DATASET_MAIN` variable is just a placeholder for `pyramidal_cells`, `interneurons`, etc., but layers are not included.
- `DATASET` variable used is a placeholder for names like `pyramidal_cells`, `pyramidal_cells/L5`, etc.

## 1. Data preparation
Scripts location: `data_preparation/`.

This stage filters out the morphologies and compiles a report with statistics about raw data. It then sends them to `m-car` tool for further morphology curation. The output morphologies are the ones used in the next stages. More on `m-car` tool, check [here](#Morphology-processing-tool).

The location of morphologies: `dvc/data/prepared/DATASET_MAIN`.
The location of raw data report: `dvc/pipelines/data_preparation/DATASET_MAIN/report_rawdata.html`.

## 2. Feature extraction
Scripts location: `feature_extraction/`.

This stage created different feature extractors depending on the dataset.
#TODO: complete when agreed

The location of morphologies and extracted features is: `dvc/data/prepared/DATASET`.

## 3. Train and evaluate
Scripts location: `train_and_evaluate/`.

This stage trains and evaluates the models with cross-validation.
The checkpoint file collects models and performances per split. It also includes the final averaged accuracy across the splits---a metric shown in the next stage.
#TODO: train model on all data, and save that one too?

The location of checkpoint file is: `dvc/models/DATASET/CHECKPOINT.chk`

## 4. Results
Scripts location: `results/`.

This stage generates HTML reports containing the summary of different model checkpoints from the previous stage (train and evaluate) per dataset.
For example, for Pyramidal Cells dataset, there is one report summarizing the performance of different models on different cortical layers.

The location of such a report is: `dvc/pipelines/results/DATASET_MAIN/results.html`.

## 5. Explanation
Scripts location: `explanation/`.

#TODO:

## Appendix

### Morphology processing tool
The [morphology processing workflow](https://bbpteam.epfl.ch/documentation/projects/morphology-processing-workflow/latest/index.html) is used to prepare and process raw morphologies to ensure they are usable by the rest of BBP codes.
Under-the-hood this tool programmatically runs [`luigi`](https://luigi.readthedocs.io/) tool, meaning there are a few files that have to be provided to the pipeline.

The main files to run the cleaning of morphologies are:
- `luigi.cfg` - more details on the supported configuration can be seen [here](https://luigi.readthedocs.io/en/stable/configuration.html),
- `logging.conf` - the configuration file for logging,
- `run_curation.sh` - the file that executes the curation phase of morphology processing,
- `dataset.csv` - this file is unique to the dataset, with the following columns:
    - `morph_name` - morphology name, e.g. `C270106A`,
    - `mtype` - morphology type, e.g. `L1_DAC`,
    - `morph_path` - path to the morphology, e.g. `path/to/file/C270106A.h5`.

Steps to perform the cleaning
1. Go to the dataset directory, e.g. `Interneurons` and make sure there is a set of `*.h5` files containing neuronal morphologies and `neurondb.dat|.xml` containing the metadata for the neurons.
2. Use and modify accordingly the `create_dataset.py` script that will generate `dataset.csv` file needed for `m-car` tool. In addition, this script also generates the `report_rawdata.pdf` that is checking for duplicated files, duplicated morphologies and gives short statistics on how many neurons are found per cell type. The `report_template.html` is just an outline of this pdf report that will be generated.
3. Run `m-car`tool with the `Curation` phase with `morphology_processing_workflow --local-scheduler Curate`. As an output, it will generate the `out_curated` directory and `report_curation.pdf`. The `out_curated` directory will contain 13 subdirectories representing the 13 stages of curating the morphologies. The final clean data is located in the path specified inside the `out_curated/curated_dataset.csv` file in the `morph_path` column.

### DVC add
Since wildcards are not supported in `dvc.yaml` files, the following files should be explicitly added with `dvc add`:
- `data/raw/DATASET_NAME`
- `data/prepared/DATASET_NAME`
