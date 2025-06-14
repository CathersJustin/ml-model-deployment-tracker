# ml-model-deployment-tracker
A Streamlit dashboard to monitor deployed machine learning models: performance tracking, drift detection, and simulated retraining.

## Setup

Install dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Logs are stored in `data/predictions_log.csv` and model metadata in `model_registry.json`.