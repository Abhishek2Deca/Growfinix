# 👥 Customer Segmentation

An unsupervised machine learning app that predicts which customer segment a
person belongs to, based on **K-Means clustering** trained on customer
demographics and purchase behavior — served through a simple Streamlit UI.

## Architecture

```
customer_segmentation.csv     Raw customer data (Age, Income, Total_Spend,
        │                     NumWebPurchases, NumStorePurchases,
        │                     NumWebVisitsMonth, Recency, ...)
        ▼
Analysis_Model.ipynb          1. Explores and cleans the data
        │                     2. Scales features with StandardScaler
        │                     3. Fits a K-Means clustering model
        │                     4. Saves the trained model + scaler to disk
        ▼
  scaler.pkl                  Saved StandardScaler — rescales new input data
        │                     to match the distribution the model was
        │                     trained on
        ▼
  kmeans_model.pkl            Saved trained K-Means clustering model
        ▼
  segmentation.py             Streamlit app: takes manual input for a single
                               customer, scales it, and predicts which
                               cluster/segment they belong to
```

**Features used for prediction:**
`Age`, `Income`, `Total_Spend`, `NumWebPurchases`, `NumStorePurchases`,
`NumWebVisitsMonth`, `Recency`

**Why this stack:**
- **K-Means** — simple, fast, interpretable clustering algorithm well suited
  to numeric customer attributes like income and spend.
- **StandardScaler** — K-Means is distance-based, so features need to be on
  a comparable scale before clustering (e.g. Income in the tens of
  thousands vs. Recency in single/double digits).
- **Pickle (.pkl) + joblib** — persists the trained model and scaler so the
  Streamlit app doesn't retrain anything on every run.
- **Streamlit** — quick interactive form for entering a customer's details
  and getting an instant segment prediction.

## Setup

1. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy joblib scikit-learn
   ```

2. **Explore / retrain the model** (optional):
   Open `Analysis_Model.ipynb` in Jupyter or VS Code to explore the data,
   tune cluster count, and regenerate `kmeans_model.pkl` / `scaler.pkl` if
   your dataset changes.

3. **Run the app**:
   ```bash
   streamlit run segmentation.py
   ```
   It'll open in your browser at `http://localhost:8501`. Enter a
   customer's Age, Income, Total Spend, purchase counts, web visits, and
   recency, then click **Predict Segment**.

## ⚠️ Known issue: model filename casing

`segmentation.py` currently loads the model as:
```python
kmeans = joblib.load('Kmeans_model.pkl')
```
but the saved file in this repo is named `kmeans_model.pkl` (lowercase
`k`). On Windows this works fine (filenames aren't case-sensitive there),
but it **will fail with a `FileNotFoundError`** on Linux, macOS, or when
deployed to Streamlit Community Cloud (which runs on Linux). Fix by
changing the load line to:
```python
kmeans = joblib.load('kmeans_model.pkl')
```
or renaming the saved file to match — just make sure both are consistent.

## Using your own data

Replace `customer_segmentation.csv` with your own customer dataset, keeping
(or remapping in the notebook) the same columns: `Age`, `Income`,
`Total_Spend`, `NumWebPurchases`, `NumStorePurchases`,
`NumWebVisitsMonth`, `Recency`. Re-run `Analysis_Model.ipynb` to regenerate
`scaler.pkl` and `kmeans_model.pkl` for the new data.

## Notes for extending this project

- **Input validation matters more than it looks**: as the code comments in
  `segmentation.py` note, using tiny input ranges (like 0–100 for Income)
  causes every input to scale to nearly the same extreme z-score,
  collapsing all predictions into a single cluster. Keep the number input
  ranges/defaults aligned with the real scale of the training data (e.g.
  Income up to ~$700k, Total Spend up to ~$3,000).
- **Cluster profiling**: right now the app only returns a cluster number.
  Consider mapping each cluster to a human-readable label (e.g. "high
  spenders, low engagement") based on the average feature values per
  cluster from the notebook.
- **Choosing cluster count (k)**: if not already done in the notebook,
  add an elbow method or silhouette score analysis to systematically pick
  the best number of clusters.
- **Alternative algorithms**: if clusters aren't clean/globular, DBSCAN or
  Gaussian Mixture Models may separate customers better than K-Means.
