# Tamil Nadu Soil & Crop Atlas

A Python-powered exploratory dashboard for 1,862 crop records across all 38 Tamil Nadu districts and 89 fields.

## How it works

- `templates/index.html` is the dashboard interface.
- `app.py` and pandas calculate district profiles, EDA summaries, correlations, completeness checks, and model predictions.
- Matplotlib generates charts on the Python server. Python also builds the clickable district-boundary map from the bundled GeoJSON.
- The **Zone-match model** tab benchmarks a most-frequent baseline, Logistic Regression, Decision Tree, Random Forest, Extra Trees, and Histogram Gradient Boosting for the source dataset’s existing `Zone_Match` Yes/No label. It selects the top grouped-validation macro-F1 model and reports held-out-district permutation importance.
- The model is evaluated with five-fold GroupKFold by district, so a district’s rows are held out together within a validation fold.

## What the model predicts

It predicts whether a district–crop row would have a `Zone_Match` value of Yes or No in this dataset. It does **not** predict actual crop success, harvest, yield, or guarantee agronomic suitability. The score measures agreement with the source dataset’s labels only.

## Run on your computer

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ml/train_zone_match.py
python app.py
```

Then open <http://localhost:5000>. On Windows, activate the environment with `.venv\\Scripts\\activate` and run the same Python commands. The training command benchmarks all listed models, selects the best by grouped-validation macro-F1, and saves the grouped-validation metrics and fitted model from the source CSV.

## Deploy from GitHub with Render

GitHub Pages serves static files and cannot run the Python analysis server. Use Render to host this Python version:

1. Upload the project contents to the root of your GitHub repository. Keep `app.py`, `requirements.txt`, `render.yaml`, `templates/`, `data/`, `ml/`, and `models/` together.
2. In Render, choose **New → Web Service**, connect the GitHub repository, and create the service. The included `render.yaml` provides the build, start, and health-check settings.
3. When deployment finishes, open the service URL. Pushes to the connected branch will redeploy it.

Render's official Flask deployment guide: <https://render.com/docs/deploy-flask>.
