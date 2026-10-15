"""Servidor HTTP liviano para la demostración interactiva en producción.
Evidencia de Aprendizaje N.º 3 - Desarrollo de Inteligencia Artificial (ISPC)
"""

import os
import sys
import json
import mimetypes
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import pandas as pd
import numpy as np

# Asegurar acceso a los módulos del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_optimization import load_model_artifact

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")

# Carga en memoria del pipeline entrenado (preprocesador + RandomForest)
print(f"[DEMO SERVER] Cargando modelo en memoria desde: {MODEL_PATH} ...")
model_pipeline = load_model_artifact(MODEL_PATH)
print("[DEMO SERVER] Modelo cargado exitosamente. Listo para recibir solicitudes.")


class DemoAppHandler(SimpleHTTPRequestHandler):
    """Manejador HTTP que sirve archivos estáticos y atiende peticiones REST de inferencia."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        # Enrutar la raíz '/' al index.html
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()
        if self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {
                "status": "healthy",
                "model": "RandomForestRegressor (Optimizado GridSearchCV)",
                "framework": "scikit-learn 1.6+",
                "features": ["rim", "width", "ratio", "Brand"]
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return
        return super().do_GET()

    def do_OPTIONS(self):
        # Soporte CORS para desarrollo local
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)

                # Validar y parsear atributos
                rim = float(data.get("rim"))
                width = float(data.get("width"))
                raw_ratio = data.get("ratio")
                ratio = None if raw_ratio is None or str(raw_ratio).strip() == "" else float(raw_ratio)
                brand = str(data.get("brand", "BRAND_A")).strip()

                # Construir DataFrame idéntico al esquema de entrenamiento
                input_df = pd.DataFrame([{
                    "rim": rim,
                    "width": width,
                    "ratio": np.nan if ratio is None else ratio,
                    "Brand": brand
                }])

                # Ejecutar inferencia a través del pipeline serializado
                pred = float(model_pipeline.predict(input_df)[0])

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

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))

            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                err_resp = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(err_resp, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, DemoAppHandler)
    print("=" * 65)
    print(f" DEMO APP LISTA EN: http://localhost:{port}")
    print(f" Directorio de interfaz estática: {STATIC_DIR}")
    print(" Presione CTRL+C para detener el servidor.")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[DEMO SERVER] Servidor detenido por el usuario.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
