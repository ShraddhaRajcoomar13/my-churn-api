from pycaret.classification import load_model, predict_model
import pandas as pd

class ChurnPredictor:
    def __init__(self, model_path="model/churn_segment_highrisk_2026.pkl"):
        print("Loading PyCaret model...")
        self.model = load_model(model_path.replace(".pkl", ""))  # PyCaret removes .pkl in load_model
        print("Model loaded.")

    def predict(self, data: dict | pd.DataFrame):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Quick validation: check for unhashable types
        for col in df.columns:
            sample = df[col].iloc[0] if not df.empty else None
            if isinstance(sample, (list, dict, set)):
                raise ValueError(
                    f"Column '{col}' contains unhashable type ({type(sample).__name__}). "
                    "All values must be scalars (number, string, bool)."
                )

        # Your existing preprocessing...
        if 'tenure' in df.columns:
            df['tenure_minmax'] = df['tenure'] / 72.0
        if 'MonthlyCharges' in df.columns:
            df['MonthlyCharges_minmax'] = df['MonthlyCharges'] / 120.0
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        result = predict_model(self.model, data=df, raw_score=True)
        result = result.rename(columns={
            'prediction_label': 'Predicted_Churn',
            'prediction_score': 'Churn_Probability'
        })

        if result['Predicted_Churn'].dtype == object:
            result['Predicted_Churn'] = result['Predicted_Churn'].map({'Yes': 1, 'No': 0})

        return result.to_dict(orient='records')