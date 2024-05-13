Contact
-------
For any inquiries or if you encounter issues with the model, please don't hesitate to get in touch. We are open to collaborating to resolve any problems and to potentially adjust the model to better suit your specific data needs.

Aaron Kieslich
Email: aaron.kieslich@oncoray.de


Usage
-----

## Installation

First, create a (conda) virtual environment like this:
```bash
conda create -n let python=3.8
```

We used the following library versions:

```bash
monai=1.1.0
seaborn=0.11.2
scipy=1.7.1
torch=1.10.0
pydicom=2.2.2
pytorch_lightning=1.9.0
```

Install the required libraries and our package
```bash
conda activate let
cd src
pip install -r requirements.txt
pip install -e .
```

Afterwards, the library should be importable from the python interpreter
```python
from let.data import LETDatasetInMemory, LETFileDataset
```



## Model training

Model training can either be performed on a local machine (preferably with a GPU and lots of main memory) or on a SLURM-powered HPC cluster.
It assumes that the above mentioned anaconda environment has been created already.
Major points to adjust in the script are the variables defined in capital letters.
E.g. `MODEL_TYPE` can be used to switch the network architecture, `CT_FILENAME` can be adjusted to use SPR CTs instead of mono-energetic CTs and similarly `LET_FILENAME` can be adjusted to switch between learning a model that predicts dose-averaged and track-averaged LET.


Navigate to the scripts directory
```bash
cd src/scripts/model_training
```

When executing on a cluster using the slurm batch system, type
```bash
# on hemera or any batch system using slurm
sbatch run_train_cv.sh
```
and 5 GPU jobs will be spawned to perform cross-validation training in parallel.

When executing the script locally, cross validation can not be performed in parallel and the variable `FOLD` has to be set manually to a number between 0 and 4. Alternatively, a single model training without cross-validation can also be performed. Then adjustements of `TRAIN_ID_FILE` and `VALID_ID_FILE` are necessary to point to files containing the patient ids used for training and (internal) validation during training.
Then execute the script via
```bash
./run_train_cv.sh
```

Apart from training the neural network, it will also perform the two evaluation steps outlined in our paper, i.e. compute RMSEs 1) within selected regions of interest between MC simulations and model predictions and 2) of NTCP models when exchanging the LET predictions from MC simulations to neural-network predictions.

## Required data and explanation of script variables
In the above script, `DATA_DIR` points to the path where the imaging data is stored. Within the given path, there needs to be a directory for each treatment plan, named using the plan id.
Within each plan's directory, preprocessed numpy arrays in the form of `*.npy` files have to be provided for the following quantities (all plans are expected to have the equal filenames and all arrays are expected to be of equal dimensionality Z x Y x X):
    - three dimensional dose distribution (as specified by `DOSE_FILENAME`)
    - the corresponding LET distribution (as specified by `LET_FILENAME`)
    - (optional) the corresponding planning CT (as specified by `CT_FILENAME`) with (unnormalized) CT numbers in HU
    - the binary segmentation masks (as specified by the `ROI_FILENAME`). Note this has to be a dict of arrays when storing to disk, where keys are ROI names provided in the `INFERENCE_ROIS` variable during inference.
    If a crop size is specified (as with CROP_SIZE), proper cropping requires the keys 'External' and 'BrainStem' (see our paper for a description of our cropping method). All other ROIs are purely required for inference and NTCP modelling purposes.

That means, the data layout is expected to be like this, assuming
```
CT_FILENAME="CT.npy"
LET_FILENAME="LETd.npy"
ROI_FILENAME="ROIs.npy"
DOSE_FILENAME="plan_dose.npy"
```

```bash
├── patient0001_plan-subseries1
│   ├── CT.npy
│   ├── LETd.npy
│   ├── plan_dose.npy
│   └── ROIs.npy
└── patient0001_plan-subseries2
    ├── CT.npy
    ├── LETd.npy
    ├── plan_dose.npy
    └── ROIs.npy
```


However, in order to create the `*.npy` files from DICOM-files in the correct structure the `src/scripts/data_preparation/interpolation.py` script can be used. The interpolation script effectively aligns dose, LET and RTSS data with the corresponding CT images. It also standardizes data to uniform voxel sizes, addressing resolution variances across sources. Additionally, the script incorporates adjusting names for ROIs through the use of an ROI renaming table. This table can standardise ROI names across different datasets, ensuring consistency in data processing and analysis. For the interpolation, paths to the specific files can be adjusted in `src/scripts/data_preparation/input_variables.py`. The interpolation script utilizes two xlsx files to facilitate the processing and analysis: the Plan Table (also specified as `PLAN_TABLE_PATH` variable, which is also required for model training and NTCP evaluation) and the ROI renaming table.

The plan table needs to have the following columns:
- plan_id :
    Identifier of the plan, same as directory names for the image data.

    Plan ids have to be in a way to start with the patient id, followed by "_" with additional arbitrary modifiers of your choice, e.g. "patient0001_plan-subseries1", "patient0001_plan-subseries2", etc...

- CTV :
    Identifier of the correct CTV name in the RTSS.

- CTV_SiB :
    Identifier of the correct name of the simultaneously integreated boost (SiB) of the treatment plan. Can be left empty, if there was no SiB present. This is needed for a correct dose normalisation.

- fxApplied :
    The number of applied radiotherapy fractions. This is needed to compute the dose per fraction in the Wedenberg RBE model.

Additionally, some NTCP models might require features that are not directly dose related, such as the presence of chemotherapy (e.g. the `NTCPFatigueG1At24m` class requires the presence of a column named `CTx`). It is your responsibility to provide this kind of information within this excel file.

The ROI renaming table should have folllowing structure:The columns lists the ROI names that the script will use for analysis. The rows specifiy possible names of the ROI as they are (or might be) defined in the RT structure sets of the DICOM files. These names may vary due to clinician preference, institutional conventions or typos.

An exemplary data sample obtained from a water phantom is available at https://rodare.hzdr.de/record/2764 (water_phantom.zip file).
Extract the zip file into a directory named `data` at the same level as this README file. Then, a notebook showing how to interpolate and load data and make predictions using a neural network is provided at `src/notebooks/phantom_example.ipynb`.

## Model limitations and future work

Our current model evaluation includes data from two centers employing proton-beam therapy devices from the same vendor (IBA) and using the same delivery technique (pencil beam scanning, PBS). This consistency might contribute to an overly optimistic assessment of the model's generalizability. The uniform treatment hardware and plan optimization strategies, which use a conventional constant RBE value of 1.1, could limit the applicability of our results to centers utilizing different treatment systems or those that employ varying treatment delivery methods, such as double scattering.

Furthermore, the study exclusively involved patients with primary brain tumors, which may restrict the transferability of our findings to other types of cancer where tumor characteristics and patient anatomy differ significantly.

Another important consideration is the potential variability in LETd distributions that could result from novel optimization strategies, despite achieving dosimetrically equivalent plans. Since voxelwise dose distributions can vary with different optimization strategies, they might also carry non-local signatures characteristic of those strategies, affecting LETd distributions.

Our future work will aim to address these limitations by incorporating a broader array of treatment devices, techniques, and tumor types to enhance the robustness and applicability of our models. We also plan to investigate the impact of different plan optimization strategies on LET distributions to further validate and refine our approach.
