# HWRS640 Assignment 4: Streamflow Prediction

This project implements a sequence model (LSTM) to predict daily streamflow using the MiniCAMELS dataset.

## Structure
HWRS640_HW4/
│
├── main.py # Command-line interface (CLI)
├── data.py # Dataset loading and dataloaders
├── model.py # LSTM model with static feature fusion
├── train.py # Training and evaluation pipeline
├── utils.py # Metrics (RMSE, MAE, NSE, KGE)
├── visualization.py # Plot generation
│
├── data/ # Dataset files
├── outputs/ # Checkpoints, metrics, figures
├── experiment_logs/ # Training logs
├── notebooks/ # Optional notebooks
├── tests/ # Optional testing scripts
│
└── README.md

---

## Setup Instructions

1. Activate environment
```bash
conda activate spyder-env

2. Install dependencies
pip install -r requirements.txt

Command Line Interface (CLI)
All functionality is accessed through main.py.

1. Summarize dataset
python main.py summarize-data

Prints:
number of basins
time span
dynamic variables
static attributes

2. Generate exploratory plots
python main.py explore-data

Outputs (in outputs/exploration/):
streamflow time series
precipitation vs streamflow
histogram of streamflow
attribute scatter plots

3. Debug dataloaders
python main.py debug-data --seq-len 60 --batch-size 32

Shows:
tensor shapes
sample structure

4. Debug model
python main.py debug-model --seq-len 60 --batch-size 32

Verifies:

input/output shapes
number of trainable parameters

5. Train model
python main.py train \
    --seq-len 60 \
    --batch-size 64 \
    --hidden-size 64 \
    --num-layers 1 \
    --dropout 0.1 \
    --learning-rate 0.001 \
    --epochs 20 \
    --output-dir outputs

This will:

train the model
save the best checkpoint
store training history

Outputs:

outputs/checkpoints/best_model.pt
outputs/metrics/training_history.json

6. Evaluate model
python main.py evaluate \
    --checkpoint outputs/checkpoints/best_model.pt

Reports:

RMSE
MAE
NSE
KGE

7. Generate plots
python main.py plot \
    --checkpoint outputs/checkpoints/best_model.pt

Outputs (in outputs/figures/):

training loss curve
NSE/KGE evolution
observed vs predicted scatter
best/worst basin plots
