# Human Activity Recognition — Multiclass Classifier

Classify what a person is doing (walking, sitting, laying, etc.) from smartphone sensor data. Built with TensorFlow/Keras on the [UCI HAR dataset](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones/data).

The dataset comes from 30 people wearing a Samsung Galaxy S II on their waist. Accelerometer and gyroscope readings were captured at 50 Hz, sliced into 2.56-second windows, and processed into 561 engineered features per sample. The task is to predict which of 6 activities the person was performing in each window.

---

## Activity Classes

| Label | Type |
|---|---|
| `WALKING` | Dynamic |
| `WALKING_UPSTAIRS` | Dynamic |
| `WALKING_DOWNSTAIRS` | Dynamic |
| `SITTING` | Static |
| `STANDING` | Static |
| `LAYING` | Static |

---

## Models

Two architectures are available, selectable via the `--model` flag:

### Feed-Forward Network (`ffn`)

A straightforward dense network. Three hidden layers (512 → 256 → 128 units), each followed by batch normalization, ReLU, and dropout. Ends with a softmax output over the 6 classes.

### 1D Convolutional Network (`cnn`)

Treats the 561-feature vector as a 1D signal and runs Conv1D layers over it to pick up local feature correlations. Three conv blocks (64 → 128 → 256 filters) with batch norm, ReLU, max pooling, and dropout, followed by global average pooling and a dense classification head.

Both models use Adam with sparse categorical cross-entropy and include early stopping + learning rate reduction on plateau out of the box.

---

## Results (Test Set)

| Metric | FFN | CNN |
|---|---|---|
| Accuracy | 93.5% | 92.0% |
| F1 (macro) | 0.935 | 0.920 |
| Precision (macro) | 0.941 | 0.923 |
| Recall (macro) | 0.933 | 0.919 |
| AUC (macro, OVR) | 0.996 | 0.996 |

Per-class breakdown, confusion matrix plot, and a full metrics JSON are saved to `logs/metrics/<model>/` after each run.

---

## Project Structure

```
multiclass/
├── src/
│   ├── train.py              # Training and evaluation entry point
│   ├── models.py             # FFN and CNN model definitions
│   ├── dataset.py            # CSV loading and tf.data pipeline
│   ├── download_dataset.py   # Kaggle download + train/val split
│   ├── metrics.py            # Confusion matrix, F1, precision, recall, AUC
│   ├── utils.py              # GPU config, seeding, Keras callbacks
│   └── logger_setup.py       # Loguru configuration
├── data/                     # Dataset CSVs (downloaded, not tracked in git)
├── saved_models/             # Trained .keras model files
├── notebooks/
│   └── EDA.ipynb             # Exploratory data analysis
├── docs/
│   └── DATASET.md            # Detailed dataset documentation
├── logs/
│   ├── metrics/              # Evaluation outputs (JSON + plots)
│   └── tensorboard/          # TensorBoard event files
├── set_env_variables.sh      # Kaggle API token + GPU library paths
├── pyproject.toml
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.12+, a Kaggle account (for downloading the dataset).

1. **Clone the repo and create a virtual environment**

   ```bash
   git clone https://github.com/Oluwadolapolasisi/multiclass
   cd multiclass
   python -m venv .venv
   source .venv/bin/activate
   ```

   Or, if you use [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Kaggle API token**

   Copy the environment script template, fill in your token, and source it:

   ```bash
   cp set_env_variables.sh set_env_var.sh
   # Edit set_env_var.sh and replace 'xxxxxxxxxxxxxx' with your Kaggle token
   source set_env_var.sh
   ```

   This also configures `LD_LIBRARY_PATH` for TensorFlow GPU support if you're using the virtual environment.

4. **Download the dataset**

   ```bash
   python src/download_dataset.py
   ```

   This pulls the data from Kaggle into `./data/` and splits `train.csv` into `train_dataset.csv` and `val_dataset.csv` (85/15 split by default).

---

## Training

Run from the project root:

```bash
# Train the feed-forward network (default)
python src/train.py --model ffn --epochs 100

# Train the 1D-CNN
python src/train.py --model cnn --epochs 100
```

Models are saved to `saved_models/har_ffn.keras` or `saved_models/har_cnn.keras` by default. You can override this with `--model_path`.

### Key flags

| Flag | Default | What it does |
|---|---|---|
| `--model` | `ffn` | Architecture: `ffn` or `cnn` |
| `--epochs` | `100` | Max training epochs |
| `--batch_size` | `64` | Batch size |
| `--learning_rate` | `0.001` | Initial learning rate (Adam) |
| `--dropout_rate` | `0.3` | Dropout after hidden layers |
| `--patience` | `10` | Early stopping patience |
| `--seed` | `42` | Random seed for reproducibility |
| `--data_dir` | `./data` | Where to find the CSV files |
| `--evaluate_only` | off | Skip training, just evaluate a saved model |

---

## Evaluation

To evaluate a previously trained model on the test set without retraining:

```bash
python src/train.py --evaluate_only --model ffn --model_path saved_models/har_ffn.keras
```

This computes the full suite of metrics (confusion matrix, per-class precision/recall/F1, AUC) and writes everything to `logs/metrics/<model>/`.

---

## TensorBoard

Training logs, histograms, and evaluation scalars are written to `logs/tensorboard/`. To view them:

```bash
tensorboard --logdir logs/tensorboard
```

Each model type gets its own subdirectory (`logs/tensorboard/ffn/`, `logs/tensorboard/cnn/`), so you can compare runs side by side.

---

## Dataset Documentation

See [`docs/DATASET.md`](docs/DATASET.md) for a detailed writeup on how the data was collected, what the sensor readings look like, how the sliding window works, and what the 561 features represent.

---

## Tech Stack

- **Python 3.12**
- **TensorFlow 2.21** — model building, training, GPU support
- **scikit-learn** — train/val split, classification metrics, AUC
- **pandas** — CSV handling
- **matplotlib** — confusion matrix plots
- **Loguru** — structured logging with daily rotation
- **kagglehub** — dataset download from Kaggle
- **uv** — dependency management (optional, pip works too)

---

## Contributors

| Name | GitHub |
|---|---|
| GREEN Emmanuel  | [@emmanuelgreen1](https://github.com/emmanuelgreen1) |
| Ipadeola Ladipo | [@rileydrizzy](https://github.com/rileydrizzy) |
| LADEPO Folaranmi | [@Fola-l](https://github.com/Fola-l) |
| LASISI Oluwadolapo | [@Oluwadolapolasisi](https://github.com/Oluwadolapolasisi) |
| MACAULAY Emmanuel | [@Oba-max22](https://github.com/Oba-max22) |
| MADUAGWUNA Onyedikachukwu  | [@lotannamoldon](https://github.com/lotannamoldon) |

