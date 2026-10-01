# 📚 Student Course Purchase Prediction

Predicts whether a student on an online learning platform will **purchase a course**, using learning-behavior data. Built during a Data Science Internship at Learn Depth™.

## Project structure

```
Course-Purchase-Prediction/
├── data/
│   └── student_course_purchase.csv     # 500 students, 5 features + target
├── notebook/
│   └── course_purchase_prediction.ipynb # EDA, charts, insights, training, evaluation
├── backend/                             # FastAPI backend
│   ├── main.py                          # REST API + serves the frontend
│   ├── train.py                         # trains model, writes model.pkl + metrics.json
│   └── model/                           # model.pkl, metrics.json
├── frontend/                            # HTML / CSS / JavaScript UI
│   ├── index.html
│   ├── style.css
│   └── script.js
├── streamlit_app/app.py                 # optional Streamlit version
├── requirements.txt
└── README.md
```

## Dataset features

| Feature | Description |
|---|---|
| age | Age of the student |
| study_hours_per_week | Hours studied weekly |
| previous_courses_completed | Courses already completed |
| platform_visits_per_month | Platform visits per month |
| assignment_completion_rate | % of assignments completed |
| **purchased_course** | **Target** (0 = No, 1 = Yes) |

## Setup

```bash
pip install -r requirements.txt
```

## Run the full web app (frontend + backend)

From the project root:

```bash
uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000** for the web app, or **http://127.0.0.1:8000/docs** for the interactive API docs.

## Run the Jupyter notebook

```bash
jupyter notebook notebook/course_purchase_prediction.ipynb
```

## Retrain the model

```bash
python backend/train.py
```

## Run the optional Streamlit app

```bash
streamlit run streamlit_app/app.py
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Predict purchase for one student |
| GET | `/api/metrics` | Model accuracy, precision, recall, F1, confusion matrix |
| GET | `/api/insights` | Feature correlations and buyer vs non-buyer averages |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":22,"study_hours_per_week":16,"previous_courses_completed":3,"platform_visits_per_month":20,"assignment_completion_rate":75}'
```

## How it works

```
Browser form (frontend) → POST /api/predict (FastAPI backend) → model.pkl (Logistic Regression) → JSON result → shown on page
```

## Key insights

1. **Study hours** is the strongest predictor of purchase (correlation ≈ 0.52).
2. **Platform visits**, **previous courses** and **assignment completion** all increase purchase likelihood.
3. **Age** has almost no effect (≈ -0.06).

## Model

- **Algorithm:** Logistic Regression (binary classification, interpretable, good for small datasets)
- **Split:** 80% train / 20% test, `random_state=42`
- **Test accuracy:** 80%

## Tech stack

Python · Pandas · NumPy · Matplotlib · Scikit-learn · Pickle · FastAPI · Uvicorn · HTML/CSS/JavaScript · Streamlit · Jupyter Notebook

## Possible improvements

- Compare with Random Forest / Gradient Boosting
- Add feature scaling and hyperparameter tuning
- Store predictions in a database
- Deploy the API (Render, Railway, Docker)
