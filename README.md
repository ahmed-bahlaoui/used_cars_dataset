# Morocco Used Cars Price Predictor & Analysis

Welcome to the Morocco Used Cars project! This project encompasses data processing, Quarto presentations for insights and analysis, inference modeling, and an interactive Streamlit user interface to predict car prices.

## Installation & Setup

This repository uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

**Note**: if you don't have uv installed on your system, please do install it and use the commands below to install the dependencies.



### 1. Install `uv`
If you haven't already installed `uv`, follow the official installation guide or run:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Sync
Clone the repository and automatically resolve dependencies by running `uv sync`:
```bash
git clone <repository-url>
cd used-cars-morocco
uv sync
```
This will automatically create a `.venv` virtual environment and install all required project packages listed in the `pyproject.toml` / `uv.lock`.
## Getting Started

Follow these instructions to run the various components of the project locally.

### Run the Prediction App
To interact with the price prediction model through the Streamlit web interface, run:
```bash
uv run streamlit run app/app.py
```
**Note**: Please run all commands in the terminal with `uv run` prefix to ensure that the correct environment is used and from the root directory of the project.

### Train the Model
If you've modified preprocessing logic or just want to retrain the underlying Gradient Boosting Regressor model, run the training pipeline:
```bash
uv run python src/train.py
```

## Presentations & Reports [Optional]
If you want, you can view and statically compile the Quarto presentation (`slides.qmd`) containing exploratory data analysis and project findings.
However you need to have quarto installed on your system.
if you don't please do install it and use the commands below to view and compile the presentation.

**Preview locally in the terminal (without opening a browser immediately):**
```bash
quarto preview claude_slides_updated.qmd --no-browser
```

**Render to static HTML:**
```bash
quarto render claude_slides_updated.qmd --to revealjs --output-dir docs --output index.html --self-contained
```

## Project Structure

```
used-cars-morocco/
├── data/
│   ├── raw/                  # Original Mendeley dataset, never touched
│   └── processed/            # Cleaned CSVs output by preprocessing
├── notebooks/
│   ├── notebook.ipynb
│
├── src/
│   ├── preprocess.py         # Reusable cleaning functions
│   └── train.py              # Model training + evaluation
├── models/                   # Saved model artifacts (.pkl, .joblib)
├── reports/
│   └── presentation.qmd      # Your Quarto file
├── app/                      # Your Streamlit or alternative app
│   └── app.py
├── requirements.txt
└── README.md
```
