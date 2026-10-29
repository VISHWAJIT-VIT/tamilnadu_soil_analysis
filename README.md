# Tamil Nadu Soil & Crop Atlas

A Python-powered exploratory dashboard for 1,862 crop records across all 38 Tamil Nadu districts and 89 fields.

## How it works

- `templates/index.html` is the complete dashboard interface.
- `app.py` uses pandas and NumPy to calculate district profiles, EDA summaries, correlations, missing-value checks, and crop/soil comparisons.
- Matplotlib generates all charts on the Python server. Python also creates the selectable district-boundary map as SVG.
- `data/master_dataset.csv` and `data/tamil-nadu-districts.geojson` are bundled so the app runs without external map tile services or an API key.
- Browser code handles interface interactions, requests the selected results, and displays Python-generated graphics. EDA calculations and charts are performed by Python.

## Run on your computer

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open <http://localhost:5000>. On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Deploy from GitHub with Render

GitHub Pages serves static files and cannot run the Python analysis server. Use Render to host this Python version:

1. Extract this project and upload its contents to the root of your GitHub repository. Keep `app.py`, `requirements.txt`, `render.yaml`, `templates/`, and `data/` together.
2. In Render, choose **New → Web Service**, connect the GitHub repository, and create the service. The included `render.yaml` provides the build, start, and health-check settings.
3. When the deployment finishes, open the service URL. Pushes to the connected branch will redeploy it.

Render's official Flask deployment guide: <https://render.com/docs/deploy-flask>.
