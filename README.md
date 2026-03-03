# F1 Qualifying Predictor

Streamlit app that predicts Formula 1 qualifying positions using a stacked ensemble (CatBoost + LightGBM + Ridge) trained on 2020–2024 data. Features driver form, circuit difficulty, and constructor trends; includes an interactive EDA dashboard and single-driver or batch CSV prediction.

**Live app:** [zainulwahaj-f1-entry-j7tdfa.streamlit.app](https://zainulwahaj-f1-entry-j7tdfa.streamlit.app/)

## Tech Stack

| Layer | Technology |
|-------|------------|
| **ML** | Scikit-Learn (StackingRegressor, Ridge), CatBoost, LightGBM |
| **Features** | category_encoders (TargetEncoder), custom driver/circuit/constructor aggregates |
| **App** | Streamlit, streamlit-option-menu |
| **Viz** | Plotly, Seaborn, Matplotlib |
| **Data** | Pandas, NumPy; Ergast API–derived F1 data (2020–2024) |

## Project Structure

```
f1/
├── entry.py              # Streamlit app (prediction UI, EDA entry)
├── pipeline.py           # Data load, preprocessing, model train/predict
├── eda.py                # EDA report and Plotly dashboards
├── f1_master_data.csv    # Master dataset (races, drivers, circuits, qualifying)
├── f1_features.csv       # Engineered features (cached)
├── qualifying_predictor_stack.pkl
├── target_encoder.pkl
├── requirements.txt
└── README.md
```

## Features

- **Stacked ensemble** — Meta-learner (Ridge) on CatBoost + LightGBM base models for qualifying position regression
- **Feature engineering** — Driver form, circuit difficulty, constructor trends, lap-time parsing, historical aggregates
- **EDA dashboard** — Four Plotly tabs: position distributions, driver/circuit performance, feature correlations
- **Prediction** — Single-driver (select driver, circuit, constructor) or batch CSV upload with progress
- **Caching** — Model and encoder loaded once; preprocessed features cached

## Local Setup

### Prerequisites

- Python 3.10+
- `f1_master_data.csv` in the project root (Ergast-derived or compatible schema)

### Install and run

```bash
git clone https://github.com/zainulwahaj/f1.git && cd f1
pip install -r requirements.txt
streamlit run entry.py
```

Open the URL shown in the terminal (default `http://localhost:8501`).

### Data

Place `f1_master_data.csv` in the repo root. The pipeline expects columns for drivers, circuits, constructors, qualifying positions, and lap times (e.g. from [Ergast API](http://ergast.com/mrd/) or an equivalent export). Running the app will build `f1_features.csv` and use the saved model if `qualifying_predictor_stack.pkl` and `target_encoder.pkl` exist; otherwise it will train and save them (or use the `models/` folder if you use that layout).

## License

MIT
