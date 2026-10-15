"""Vercel Serverless Function - Inferencia de Costo de Neumáticos
Desarrollo de Inteligencia Artificial - ISPC (TeamYear3)
"""

import os
import sys
import json
from http.server import BaseHTTPRequestHandler
import numpy as np
import pandas as pd
import joblib

# Ubicación del modelo (compatible tanto con despliegue Vercel como entorno local)
CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.joblib"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "best_model.joblib"),
    os.path.join(os.getcwd(), "models", "best_model.joblib"),
    os.path.join(os.getcwd(), "api", "best_model.joblib")
]

MODEL = None

def get_model():
    """Carga y cachea el pipeline en memoria para reutilización entre llamadas serverless warm."""
    global MODEL
    if MODEL is None:
        model_path = None
        for path in CANDIDATE_PATHS:
            if os.path.exists(path):
                model_path = path
                break
        if not model_path:
            raise FileNotFoundError(f"No se encontró el modelo en ninguna de las rutas: {CANDIDATE_PATHS}")
        MODEL = joblib.load(model_path)
    return MODEL


class handler(BaseHTTPRequestHandler):
    """Manejador Serverless nativo para Vercel (@vercel/python)."""

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        """Soporte para preflight CORS."""
        self._set_headers(200)

    def do_GET(self):
        """Endpoint de verificación de salud."""
        self._set_headers(200)
        resp = {
            "status": "healthy",
            "runtime": "Vercel Python Serverless",
            "model": "Random Forest Regressor (200 árboles)",
            "framework": "scikit-learn 1.6+",
            "features": ["rim", "width", "ratio", "Brand"]
        }
        self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        """Endpoint de inferencia predictiva."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            # Parsear variables con validación
            rim = float(data.get("rim"))
            width = float(data.get("width"))
            raw_ratio = data.get("ratio")
            ratio = None if raw_ratio is None or str(raw_ratio).strip() == "" else float(raw_ratio)
            brand = str(data.get("brand", "BRAND_A")).strip()

            input_df = pd.DataFrame([{
                "rim": rim,
                "width": width,
                "ratio": np.nan if ratio is None else ratio,
                "Brand": brand
            }])

            pipeline = get_model()
            pred = float(pipeline.predict(input_df)[0])

            response_data = {
                "status": "success",
                "input": {
                    "rim": rim,
                    "width": width,
                    "ratio": ratio if ratio is not None else "Imputado automáticamente (mediana)",
                    "brand": brand
                },
                "predicted_cost_inr": round(pred, 2),
                "formatted_cost": f"${pred:,.2f} INR",
                "model_info": {
                    "algorithm": "Random Forest Regressor (200 árboles)",
                    "selection_stage": "Ganador 5-Fold Cross-Validation (ISPC Evidencia 3)",
                    "cv_rmse_inr": 1381.01,
                    "test_r2": 0.8446
                }
            }

            self._set_headers(200)
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))

        except Exception as e:
            self._set_headers(400)
            err_resp = {"status": "error", "message": str(e)}
            self.wfile.write(json.dumps(err_resp, ensure_ascii=False).encode("utf-8"))
