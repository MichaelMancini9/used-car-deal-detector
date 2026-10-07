# Used Car Deal Detector

A machine learning model that estimates the fair price of a used car and flags listings priced unusually low, while separating real bargains from listings that are cheap for a reason.

**Example: a "Strong" deal from the test set**

> **2021 Toyota Tacoma TRD Sport, 9,512 miles**, listed at **$33,497**
> Expected price **$39,396**, fair range **$37,041 – $42,511**
> About **$5,900 (15%) below** what similar trucks sell for

---

## Results at a glance

| | |
|---|---|
| Listings analysed | 748,817 (after cleaning 762,091 cars.com listings) |
| Pricing error (MAPE) | **7.7%**, down from 12.1% for a comparable-sales baseline |
| Typical error | about **$1,400** per car |
| Price-range coverage | **79.9%** of real prices fall inside the predicted 80% range |
| Strong deals found | 867 of 149,764 test listings (**about 0.6%**) |

---

## Data

[Used Cars Dataset](https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset) by Andrei Novikov on Kaggle (CC0 public domain): 762,091 dealer listings scraped from cars.com in **April 2023**, with make, model/trim, year, mileage, drivetrain, fuel type, MPG, accident history, number of owners, seller rating, price drops and price.

The data isn't included in this repo. To run the notebooks, download `used_cars.csv` from Kaggle and save it as `data/cars.csv`.

---

## Approach

| Notebook | What it does |
|---|---|
| `01_explore` | Explore the data and find errors: a ~$1 billion listing, $1 placeholder prices, 1.1 million-mile cars, 1915 Model Ts |
| `02_clean` | Fix and simplify the data (details below) and save `cars_clean.parquet` |
| `03_baseline` | A simple benchmark: the median price of the same make/model/year |
| `04_model_comparison` | Ridge regression, Random Forest, CatBoost and LightGBM on identical data |
| `05_final_model` | Train LightGBM, check feature importance, and analyse where it makes mistakes |
| `06_deals` | Predict a fair price **range** for each car, calibrate it, and flag and rank deals |

### Cleaning highlights
- Kept prices between $2k and $200k and mileage up to 400k; removed placeholder mileage on older cars and duplicate listings
- Merged ~1,300 transmission spellings and dozens of fuel-type and drivetrain variants into clean categories with keyword matching
- Found that a fuel type coded **"B"** (1,442 rows) was entirely Toyota hybrids, and mapped it accordingly
- Filled missing MPG from identical make/model/year cars. The ~62k that stayed blank turned out to be mostly **heavy-duty pickups** (Ram 2500/3500, F-250/350, Silverado/Sierra HD), which aren't EPA-rated, plus electric vehicles
- Treated missing accident and ownership history as **"Unknown"**, not "No", since buyers price unknown history differently
- Left owner ratings blank where a model had **no reviews**, since "unrated" isn't the same as "average"

### Model comparison
All models use the same features and the same test cars, and predict log(price).

| Model | MAE | MAPE | Median error |
|---|---|---|---|
| **LightGBM** | **$2,219** | **7.7%** | **$1,421** |
| CatBoost | $2,486 | 8.2% | $1,526 |
| Random Forest (100k sample) | $3,112 | 10.4% | $1,857 |
| Ridge regression | $3,333 | 11.1% | $1,940 |
| Baseline (make/model/year median) | $3,070 | 12.1% | $2,004 |

Ridge regression did **worse than the simple baseline** in dollar terms. A linear model assumes every car loses value the same way, but a Civic and a Porsche depreciate very differently. LightGBM was chosen for accuracy and speed: CatBoost took over 20 minutes to train, compared with a few minutes for LightGBM.

---

## Key findings

**1. Age and mileage drive about half of the price prediction.** Feature importance: age 29%, mileage 19%, model 16%. Surprisingly, highway MPG ranked 4th (12%). That isn't because buyers pay for fuel economy: MPG acts as a stand-in for **vehicle type** (heavy trucks and V8s vs. economy cars and hybrids).

**2. Cheap cars are the hardest to price.**

| Price band | Average error |
|---|---|
| Under $10k | 20.4% |
| $10k – $20k | 9.5% |
| $20k – $35k | 6.4% |
| $35k – $60k | 5.9% |
| Over $60k | 7.3% |

Below $10k, condition (rust, maintenance, mechanical problems) drives price, and it isn't in the data. Makes with wide trim and option ranges (Dodge, Ford and Chevrolet trucks, Porsche) are also harder.

**3. That is why deals are judged against a range, not a fixed discount.** Being 15% below expected is normal noise for a $7k car but unusual for a $30k car. Two extra LightGBM models predict the 10th and 90th percentile price, and the ranges are calibrated on held-out data (conformalized quantile regression) so 80% of real prices fall inside. The ranges widen automatically where the model is less certain: about 49% of the expected price for cars under $10k, versus about 17% for cars over $35k.

**4. The biggest discounts are usually not the best deals.** The largest raw discounts were dominated by classic cars, placeholder prices (a $2,500 "price" appeared repeatedly) and accident-damaged vehicles. Flagged cars are therefore labelled before ranking:

| Label | Rule |
|---|---|
| Classic (out of scope) | Model year before 2000 |
| Likely data error | More than 50% below expected |
| Check mileage | 5+ years old and under 1,500 miles a year |
| Accident history | Accident or damage reported |
| Possible deal | Everything else |

**5. Possible deals are ranked by a deal score**, defined as how far below the bottom of its fair range a car is, relative to the range's width:

| Tier | Deal score | Cars | Median savings | Previously price-dropped |
|---|---|---|---|---|
| Slight | < 0.25 | 8,007 | $2,899 | 62% |
| Good | 0.25 – 0.5 | 2,182 | $3,932 | 62% |
| **Strong** | 0.5 – 1.0 | **867** | **$4,979** | 55% |
| Review | > 1.0 | 140 | $6,852 | 46% |
| *All test listings* | | | | *54%* |

Modest deals are often listings the seller has **already reduced**. The largest discounts usually weren't: these cars were listed low from the start, which can mean a genuine bargain or a problem the data can't see. That's why extreme cases go to **Review** instead of the top of the list. Typical Strong deals are 2017–2022 mainstream vehicles listed 15–25% below expected.

---

## Limitations

- **The model can't see condition, title status, options or undisclosed damage.** Flagged listings are leads to investigate, not guaranteed bargains.
- **A single snapshot from April 2023**, during unusually high used car prices. The model shows what was fair then, not what's fair today.
- **Dealer listings only**: no private sales, where many real bargains are found.
- **Trim is part of the model name** (e.g. "RAV4 Hybrid XLE"), which splits one vehicle into many small groups.
- The validation set was used both for early stopping and for calibrating the ranges, so the calibration isn't fully independent.

## Next steps

- Separate model and trim; extract engine features (displacement, cylinders, turbo) from the raw text
- Per-car explanations with SHAP values (e.g. *"expected $37k mainly because it's 2 years old and 4Runners hold value"*)
- A Streamlit app: enter a car and asking price, get a fair range and a deal verdict
- Add private-sale listings

---

## How to run

Requires Python 3.12+.

```bash
git clone https://github.com/MichaelMancini9/used-car-deal-detector.git
cd used-car-deal-detector
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

1. Download the dataset from Kaggle and save it as `data/cars.csv`
2. Run the notebooks in order, `01` to `06`. Notebook `02` creates `data/cars_clean.parquet`, which the later notebooks use. `06` saves its trained models to `models/` so they only train once.

## Project structure

```
used-car-deal-detector/
├── data/           # raw and cleaned data (not tracked by git)
├── models/         # saved models (not tracked by git)
├── notebooks/
│   ├── 01_explore.ipynb
│   ├── 02_clean.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_final_model.ipynb
│   └── 06_deals.ipynb
├── requirements.txt
└── README.md
```

**Tools:** Python, pandas, scikit-learn, LightGBM, CatBoost, matplotlib, Jupyter
