# Blazar Classification

App for **Blazar Classes of Unknown type** inference using:

- **Bias Initialization**
- **Greedy Supervised**

Both models share the same preprocessing flow used in `/Users/srinadb/Downloads/BCUFinal/Data_Preperation.ipynb`:

- feature order: `PL_Index, nu_syn, LP_Index, Pivot_Energy, Frac_Variability, Variability_Index, nuFnu_syn`
- optional automatic log handling for uploaded raw files
- standardization from the saved training split
- conformal prediction sets using the saved qhat files

## Structure

- `/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/inference.py`
  Shared preprocessing, model loading, and conformal inference.
- `/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/streamlit_app.py`
  Public-friendly one-page Streamlit app.
- `/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/api.py`
  FastAPI backend for the React frontend.
- `/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/frontend`
  Single-page React interface.

## Local Streamlit run

```bash
pip install -r /Users/srinadb/Downloads/BCUFinal/blazar_classification_app/requirements.txt
streamlit run /Users/srinadb/Downloads/BCUFinal/blazar_classification_app/streamlit_app.py
```

## Local React + API run

Backend:

```bash
pip install -r /Users/srinadb/Downloads/BCUFinal/blazar_classification_app/requirements.txt
uvicorn blazar_classification_app.api:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd /Users/srinadb/Downloads/BCUFinal/blazar_classification_app/frontend
npm install
npm run dev
```

The React app reads `VITE_API_BASE_URL` and defaults to `http://localhost:8000`.

## Streamlit Community Cloud deployment

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Use `/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/streamlit_app.py` as the main file.
4. Make sure the Python dependencies from `/Users/srinadb/Downloads/BCUFinal/blazar_classification_app/requirements.txt` are available at deploy time.
   If the platform expects a repo-root `requirements.txt`, copy this file to the repo root before deploying.
5. Redeploy.

## Upload expectations

- `.csv`
  - either named columns matching the 7 feature names
  - or the first 7 columns in the notebook feature order
- `.npy`
  - shape `(7,)` or `(n, 7)`

The app supports:

- `Auto detect`
- `Raw features`
- `Already log-transformed`

Manual sliders use the transformed feature space derived from the saved BCU feature table so the layout stays compact and stable on one page.
