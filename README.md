
# Creep Curve Forecasting in High-Temperature Alloys

This repository contains the experimental creep datasets and Python code associated with the manuscript:

**Early Creep Deformation Improves Physics-Guided Machine Learning Forecasts of Creep Trajectories in High-Temperature Alloys**

## Dataset

The dataset consists of 333 constant-load creep tests from three high-temperature alloys:

| Material | Number of tests |
|---|---:|
| Alloy 617 | 54 |
| Grade 91 steel | 225 |
| Wrought Haynes 282 | 54 |
| **Total** | **333** |

The datasets contain time-dependent creep strain measurements under different temperatures and applied stresses. The data are organized into three folders under `data/`:

- `A617/`: Alloy 617
- `Grade 91/`: Grade 91 steel
- `HT/`: Wrought Haynes 282

## Code

The main script, `master_model.py`, implements the creep-curve forecasting framework described in the manuscript.

The workflow includes:

1. **Data preprocessing:** Importing and processing experimental creep curves.
2. **Theta-projection fitting:** Fitting creep trajectories and extracting normalized curve descriptors.
3. **Creep forecasting:** Predicting full creep trajectories using loading conditions and partially observed creep measurements.
4. **Model evaluation:** Comparing forecasting approaches using grouped cross-validation, prediction errors, and uncertainty quantification.

## How to Run

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn openpyxl
```

Run the complete analysis:

```bash
python master_model.py --stage all
```

Alternatively, run individual stages:

```bash
python master_model.py --stage theta-fit
python master_model.py --stage forecast
```

To specify early-observation fractions:

```bash
python master_model.py --reveals 0.20 0.30
```

The analysis outputs are saved in `master_model_output/`.

**Note:** The script must be configured to read the datasets from the repository's `data/` directory.

## Citation

If you use this repository, please cite the associated manuscript and the archived repository DOI (to be added following Zenodo deposition).

