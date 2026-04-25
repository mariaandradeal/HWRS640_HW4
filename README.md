# HWRS640 Assignment 4: Streamflow Prediction

This project implements an LSTM-based sequence model to predict daily streamflow using the MiniCAMELS dataset.

---

## Project Structure


HWRS640_HW4/
│
├── main.py # Command-line interface (CLI)
├── data.py # Dataset loading and preprocessing
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

##  Setup Instructions

### 1. Activate environment
```bash
conda activate spyder-env

2. Install dependencies
pip install -r requirements.txt
🚀 Command Line Interface (CLI)

All functionality is accessed through:
python main.py <command>

1. Summarize dataset
python main.py summarize-data

Prints:
number of basins
time span
dynamic variables
static attributes

2. Generate exploratory plots
python main.py explore-data --n-basins 4

Description
Randomly selects N basins
Generates hydrographs (precipitation + streamflow)
Plots distributions of static basin attributes
Outputs (in outputs/exploration/)
outputs/exploration/
├── hydrographs/
│   ├── basin_XXXXX_hydrograph.png
│   ├── basin_XXXXX_hydrograph.png
│   └── ...
└── attributes/
    └── static_attribute_histograms.png

3. Train model
python main.py train 
    --seq-len 120 ^
    --batch-size 128 ^
    --hidden-size 128 ^
    --num-layers 1 ^
    --dropout 0.1 ^
    --learning-rate 0.0005 ^
    --epochs 20 ^
    --output-dir outputs

This will:
train the LSTM model
save the best checkpoint
store training history
Outputs:
outputs/checkpoints/best_model.pt
outputs/metrics/training_history.json

4. Evaluate model
python main.py evaluate ^
--checkpoint outputs\checkpoints\best_model.pt

Reports:
RMSE
MAE
NSE
KGE

5. Generate evaluation plots
python main.py plot \
    --checkpoint outputs/checkpoints/best_model.pt



Outputs (in outputs/figures/)
training loss curve
NSE evolution
observed vs predicted scatter
best and worst basin hydrographs
NSE ranking and spatial analysis

Notes
The model uses both dynamic meteorological inputs and static basin attributes
Target variable is daily streamflow (qobs)
Data is normalized using training statistics
Log-transform is applied to streamflow during training