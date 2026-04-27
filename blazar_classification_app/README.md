# Blazar Classification

This app predicts `BLL` and `FSRQ` probabilities for blazar sources using:

- Bias Initialization
- Greedy Supervised

It also shows conformal prediction sets for both models.

## Input

The app accepts:

- `.csv`
- `.npy`

Feature order must match:

`PL_Index, nu_syn, LP_Index, Pivot_Energy, Frac_Variability, Variability_Index, nuFnu_syn`

If uploaded values are raw, the app can apply the same log transforms and standardization used in [Data_Preperation.ipynb](/Users/srinadb/Downloads/BCUFinal/Data_Preperation.ipynb).

## Run with Streamlit

```bash
pip install -r /Users/srinadb/Downloads/BCUFinal/blazar_classification_app/requirements.txt
streamlit run /Users/srinadb/Downloads/BCUFinal/blazar_classification_app/streamlit_app.py
```

## Optional React frontend

Backend:

```bash
uvicorn blazar_classification_app.api:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd /Users/srinadb/Downloads/BCUFinal/blazar_classification_app/frontend
npm install
npm run dev
```

## Main files

- [streamlit_app.py](/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/streamlit_app.py)
- [api.py](/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/api.py)
- [inference.py](/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/inference.py)
