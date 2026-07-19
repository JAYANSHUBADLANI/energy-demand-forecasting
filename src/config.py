"""Central configuration: paths, dataset constants, model settings."""

from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_DIR = ROOT / "models"

PROCESSED_FILE = DATA_PROCESSED / "energy_weather_hourly.csv"
FEATURES_FILE = DATA_PROCESSED / "features.csv"

# ------------------------------------------------------------- raw data ----
ENERGY_CSV = "energy_dataset.csv"
WEATHER_CSV = "weather_features.csv"
KAGGLE_DATASET = "nicholasjhana/energy-consumption-generation-prices-and-weather"

TARGET = "total load actual"
CITIES = ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"]
LOCAL_TZ = "Europe/Madrid"  # calendar features & holidays use local time

# Weather columns averaged across the five cities
WEATHER_NUMERIC = ["temp", "pressure", "humidity", "wind_speed", "clouds_all"]

# ------------------------------------------------------------ modelling ----
HORIZON = 24          # day-ahead: 24 hourly steps
TEST_WEEKS = 12       # rolling-origin backtest window
LGBM_REFIT_DAYS = 7   # refit cadence inside the backtest
SARIMAX_REFIT_DAYS = 28
RANDOM_STATE = 42

LAG_HOURS = [24, 48, 168]
ROLL_WINDOWS = [24, 168]  # applied after a 24 h shift so they are known day-ahead

QUANTILES = [0.10, 0.50, 0.90]

LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    verbose=-1,
)


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (DATA_RAW, DATA_PROCESSED, REPORTS, FIGURES, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
