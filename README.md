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
quarto preview slide_deck --no-browser
```

**Render to static HTML:**
```bash
quarto render slide_deck.qmd --to revealjs --output-dir docs --output index.html --self-contained
```

## Project Structure


```
used_cars_dataset
├─ app
│  └─ app.py
├─ cars_dataframe.csv
├─ cleaned_data.csv
├─ cleaned_data_for_real.csv
├─ data
│  └─ processed
│     ├─ cleaned_data.csv
│     ├─ cleaned_data_final.csv
│     └─ cleaned_data_for_real.csv
├─ docs
│  └─ index.html
├─ evaluation
├─ figures
│  ├─ fig_01_missing_values_by_column.png
│  ├─ fig_02_price_distribution_up_to_95th_percentile.png
│  ├─ fig_03_brands_by_median_price_min_100_listings.png
│  ├─ fig_04_median_price_by_year.png
│  ├─ fig_05_median_price_by_transmission_type.png
│  ├─ fig_06_median_price_by_car_condition.png
│  ├─ fig_07_top_5_most_listed_cars.png
│  ├─ fig_08_old_cars_vs_new_cars_in_the_market.png
│  ├─ fig_09_gearbox_type_distribution.png
│  ├─ fig_10_fuel_type_distribution.png
│  ├─ fig_11_top_10_locations_by_number_of_listings.png
│  ├─ fig_12.png
│  ├─ fig_13_median_price_by_car_origin.png
│  ├─ fig_14_median_price_by_first_owner_status.png
│  ├─ fig_15_top_20_equipment_features.png
│  ├─ fig_16_mean_price_vs_number_of_features.png
│  ├─ fig_16_mean_price_vs_number_of_features_binned.png
│  ├─ fig_16_price_vs_number_of_equipment_features.png
│  ├─ fig_17.png
│  ├─ fig_17_correlation_matrix.png
│  └─ model_comparison.png
├─ functions
├─ logo_ensam.png
├─ models
│  ├─ car_price_model.json
│  ├─ gbr_pipeline.joblib
│  └─ preprocessor.joblib
├─ Moroccan Used Car Price Prediction.pdf
├─ old.qmd
├─ preprocessing
│  └─ notebooks
│     ├─ car_price_model.json
│     ├─ cleaned_data_for_real.csv
│     ├─ cleaned_data_ready_for_ml.csv
│     ├─ figures
│     │  ├─ fig_01_missing_values_by_column.png
│     │  ├─ fig_02_price_distribution_up_to_95th_percentile.png
│     │  ├─ fig_03_brands_by_median_price_min_100_listings.png
│     │  ├─ fig_04_median_price_by_year.png
│     │  ├─ fig_05_median_price_by_transmission_type.png
│     │  ├─ fig_06_median_price_by_car_condition.png
│     │  ├─ fig_07_top_5_most_listed_cars.png
│     │  ├─ fig_08_old_cars_vs_new_cars_in_the_market.png
│     │  ├─ fig_09_gearbox_type_distribution.png
│     │  ├─ fig_10_fuel_type_distribution.png
│     │  ├─ fig_11_top_10_locations_by_number_of_listings.png
│     │  ├─ fig_12.png
│     │  ├─ fig_13_median_price_by_car_origin.png
│     │  ├─ fig_14_median_price_by_first_owner_status.png
│     │  ├─ fig_15_top_20_equipment_features.png
│     │  ├─ fig_16_mean_price_vs_number_of_features.png
│     │  └─ fig_17_correlation_matrix.png
│     ├─ notebook.ipynb
│     └─ train_model.ipynb
├─ presentation.html
├─ pyproject.toml
├─ README.md
├─ reports
│  └─ index.html
├─ slide-deck.qmd
├─ slides.html
├─ slides.qmd
├─ slide_deck.qmd
├─ slide_deck_to_pdf.qmd
├─ src
│  ├─ preprocess.py
│  ├─ train.py
│  └─ train_xgboost.py
└─ uv.lock
```