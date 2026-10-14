## 📋 Overview

This API provides real-time churn predictions by wrapping a `ChurnPredictor` model loader. It is designed to be lightweight, easy to deploy, and suitable for integration into CRM, marketing automation, or retention workflows.

### Key Features

- 🚀 **FastAPI-powered** — automatic OpenAPI/Swagger docs, async-ready
- 🔄 **Batch & single predictions** — flexible input formats
- 🧩 **Dynamic model loading** — bypasses import path issues via `importlib`
- ❤️ **Health check endpoint** — for orchestration & monitoring
- 🛡️ **Robust error handling** — returns structured HTTP 500 errors on failure

---

## 🗂️ Project Structure

```
.
├── main.py              # FastAPI application (this file)
├── model_loader.py      # Contains the ChurnPredictor class
├── requirements.txt     # Python dependencies
└── README.md
```

> `model_loader.py` must be in the same directory as `main.py` and must expose a `ChurnPredictor` class with a `.predict(data)` method.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
fastapi
uvicorn
pydantic
```

---

## ▶️ Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:

- **Interactive docs (Swagger UI):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

---

## 🔌 API Endpoints

### `GET /health`
Returns service status.

**Response**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### `POST /predict`
Batch prediction — accepts a list of customer records.

**Request Body**
```json
{
  "data": [
    { "tenure": 12, "monthly_charges": 70.5, "contract": "Month-to-month" },
    { "tenure": 3,  "monthly_charges": 95.0, "contract": "One year" }
  ]
}
```

**Response**
```json
{
  "predictions": [1, 0]
}
```

---

### `POST /predict/single`
Single-record prediction.

**Request Body**
```json
{
  "tenure": 12,
  "monthly_charges": 70.5,
  "contract": "Month-to-month"
}
```

**Response**
```json
{
  "prediction": 1
}
```

---

## 🧠 Model Loader Contract

`model_loader.py` must expose:

```python
class ChurnPredictor:
    def predict(self, data):
        """
        Args:
            data: A single dict OR a list of dicts representing customer records.
        Returns:
            A list of predictions (one per record).
        """
```

The API calls `predictor.predict(request.data)` for batch and `predictor.predict(item)` for single, expecting a list in both cases.

---

## 🛡️ Error Handling

All prediction failures return:

```json
{
  "detail": "Prediction failed: <error message>"
}
```

with HTTP status `500`.

---

## 🐳 Deployment Tips

- **Docker**: Expose port `8000` and use `uvicorn main:app --host 0.0.0.0`.
- **Production**: Use `gunicorn -k uvicorn.workers.UvicornWorker main:app -w 4`.
- **Scaling**: The model is loaded once at startup — fork-safe for multi-worker deployments.
- **Monitoring**: Poll `/health` from your orchestrator (Kubernetes, ECS, etc.).

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙌 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
```
