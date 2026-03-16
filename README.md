
 # Student Exam Score Predictor 🎯
### An end-to-end machine learning system — built the way it's done in the industry.

> *"Can a student's background predict how well they'll perform in math?"*  
> That's the question this project tries to answer — with real data, real pipelines, and a real deployed web app.

---

## What This Project Is About

Most ML projects stop at the notebook. Train a model, print accuracy, done.

This one doesn't.

This project treats the entire ML workflow as a **software engineering problem** — with modular components, custom pipelines, automated model selection, proper logging, exception handling, and a live web app deployed to the cloud via CI/CD.

It predicts a student's **math score** (0–100) given information about their background, parental education, lunch type, and how they performed in reading and writing. Trained on 1,000 student records, it benchmarks **10 different models** and automatically picks the best one.

---

## Live Demo

🌐 **Deployed on Azure App Service** — every push to `main` auto-deploys via GitHub Actions.

Try it yourself: enter a student's details → get a predicted math score instantly.

---

## The Dataset

**1,000 students. 8 columns. 1 target.**

| Feature | Type | What it captures |
|---|---|---|
| `gender` | Categorical | Male / Female |
| `race_ethnicity` | Categorical | Groups A through E |
| `parental_level_of_education` | Categorical | From high school to master's degree |
| `lunch` | Categorical | Standard vs. free/reduced (a socioeconomic proxy) |
| `test_preparation_course` | Categorical | Did the student complete a prep course? |
| `reading_score` | Numerical | Score out of 100 |
| `writing_score` | Numerical | Score out of 100 |
| `math_score` ⭐ | **Target** | What we're predicting |

**Score distributions (across all 1,000 students):**

|  | Math | Reading | Writing |
|---|---|---|---|
| Average | 66.1 | 69.2 | 68.1 |
| Std Dev | 15.2 | 14.6 | 15.2 |
| Lowest | 0 | 17 | 10 |
| Highest | 100 | 100 | 100 |

---

## Model Results 📊

All models were trained with an automated evaluation loop. Here's how they ranked on the **test set** (20% holdout, never seen during training):

| Rank | Model | R² Score | MAE | RMSE |
|:---:|---|:---:|:---:|:---:|
| 🥇 | **Ridge Regression** | **88.06%** | **4.21** | **5.39** |
| 🥈 | Linear Regression | 88.04% | 4.21 | 5.39 |
| 🥉 | Gradient Boosting | 87.22% | 4.30 | 5.58 |
| 4 | Lasso Regression | 85.64% | 4.63 | 5.91 |
| 5 | AdaBoost | 85.16% | 4.68 | 6.01 |
| 6 | Random Forest | 85.04% | 4.70 | 6.03 |
| 7 | XGBoost | evaluated via GridSearchCV | — | — |
| 8 | CatBoost | evaluated via GridSearchCV | — | — |
| 9 | Decision Tree | 71.94% | 6.54 | 8.26 |
| 10 | K-Neighbors | 47.56% | 8.69 | 11.30 |

> **R² of 88%** means the model explains 88% of the variance in student math scores — a strong result for tabular data of this size.
> The best model is **automatically selected and saved** to `artifacts/model.pkl` — no manual picking.

---

## Architecture — How It's Actually Built

This isn't a single script. It's a system.

```
stud.csv  (raw data)
    │
    ▼
┌─────────────────────┐
│   Data Ingestion    │  Reads CSV → splits train/test → saves to artifacts/
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────┐
│   Data Transformation    │
│                          │
│  Numerical features:     │
│  Impute (median)         │
│  → StandardScaler        │
│                          │
│  Categorical features:   │
│  Impute (most frequent)  │
│  → OneHotEncoder         │
│  → StandardScaler        │
│                          │
│  Combined via            │
│  ColumnTransformer       │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│     Model Trainer        │  GridSearchCV on 10 models → picks best by R²
│                          │  Saves model.pkl + preprocessor.pkl
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│   Prediction Pipeline    │  Loads saved artifacts → transforms input → predicts
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│     Flask Web App        │  User fills form → pipeline → returns score
└──────────────────────────┘
          │
          ▼
   Azure App Service
   (auto-deployed via GitHub Actions on every push to main)
```

---

## Project Structure

```
mlproject/
│
├── src/                              # All source code lives here
│   ├── components/
│   │   ├── data_ingestion.py         # Reads raw data, creates train/test split
│   │   ├── data_transformation.py   # Builds and fits the preprocessing pipeline
│   │   └── model_trainer.py         # Trains all models, GridSearchCV, saves best
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py        # Orchestrates the full training flow
│   │   └── predict_pipeline.py      # Handles inference for new inputs
│   │
│   ├── exception.py                 # Custom exception class with detailed tracebacks
│   ├── logger.py                    # Timestamped logging to file on every run
│   └── utils.py                     # Shared helpers: save_object, load_object, evaluate_models
│
├── artifacts/                       # Auto-generated during training
│   ├── data.csv                     # Full raw dataset
│   ├── train.csv / test.csv         # 80/20 split
│   ├── model.pkl                    # Best trained model (serialized)
│   └── preprocessor.pkl            # Fitted preprocessing pipeline
│
├── notebook/
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb    # Exploratory data analysis
│   └── 2. MODEL TRAINING.ipynb             # Model benchmarking experiments
│
├── templates/
│   ├── index.html                   # Landing page
│   └── home.html                    # Prediction form UI
│
├── .github/workflows/
│   └── main_studentssperformance3.yml   # CI/CD → Azure
│
├── app.py                           # Flask entry point
├── setup.py                         # Makes src/ an installable Python package
└── requirements.txt
```

---

## Run It Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/DhyaneshV/mlproject.git
cd mlproject
```

**Step 2 — Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Train the model** *(runs the full pipeline end-to-end)*
```bash
python src/components/data_ingestion.py
```
This auto-generates `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

**Step 5 — Launch the web app**
```bash
python app.py
```
Open `http://localhost:5000`, fill in the form, and get a predicted math score.

---

## What Makes This Different From a Basic ML Project

| Basic ML Project | This Project |
|---|---|
| Single notebook | Modular components — each with a single responsibility |
| Manual preprocessing | `sklearn` Pipeline — no data leakage, consistent transforms |
| Train one model | Auto-benchmark 10 models with GridSearchCV, pick the best |
| Print accuracy | R², MAE, RMSE tracked for every model |
| No error handling | Custom exception class with file/line-level tracebacks |
| No logging | Timestamped logs written to file on every run |
| Run locally only | Deployed to Azure via CI/CD — live web app |
| Hard-coded paths | Config dataclasses — change one thing, it propagates |

---

## EDA Highlights

Full exploratory analysis was done before modelling (see `notebook/1. EDA STUDENT PERFORMANCE.ipynb`). Key findings:

- Students who completed the **test preparation course** scored consistently higher across all three subjects
- **Parental level of education** had a notable positive correlation with student performance
- **Lunch type** (a socioeconomic indicator) showed a clear split in average scores — standard lunch students outperformed free/reduced lunch students across the board
- Reading and writing scores were highly correlated with math scores, making them the strongest predictors in the model

---

## Tech Stack

```
ML & Data       →  scikit-learn, XGBoost, CatBoost, Pandas, NumPy
Web App         →  Flask
Visualisation   →  Matplotlib, Seaborn (EDA notebook)
Deployment      →  Azure App Service
CI/CD           →  GitHub Actions
Packaging       →  setuptools (src/ is an installable Python package)
```

---

## CI/CD Pipeline

Every push to `main` triggers the GitHub Actions workflow which:

1. Spins up a clean Ubuntu environment with Python 3.7
2. Creates a virtual environment and installs all dependencies
3. Packages the application
4. Deploys directly to **Azure App Service** (`studentssperformance3`)

No manual deployment steps. Push code → it's live.

---

## Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
catboost
xgboost
Flask
```

```bash
pip install -r requirements.txt
```

---

## Author

**Dhyanesh V**

Built this to understand how ML systems are actually structured in production — not just how to train a model, but how to build something maintainable, deployable, and understandable by someone other than yourself.

[GitHub →](https://github.com/DhyaneshV)