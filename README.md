commands to preview the presentation:

quarto preview slides.qmd --no-browser

commands to render the presentation:

quarto render slides.qmd --to revealjs --output-dir docs --output index.html --self-contained


## Project structure
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