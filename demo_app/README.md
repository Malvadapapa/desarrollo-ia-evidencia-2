# NeumaPredict AI — Aplicación de Demostración en Producción

Esta carpeta contiene una implementación desacoplada y autocontenida para la puesta en producción y prueba interactiva del modelo predictivo desarrollado en la **Evidencia de Aprendizaje N.º 3** (ISPC).

---

## 📁 Estructura de Componentes

```
demo_app/
├── server.py             # Servidor HTTP liviano en Python (REST API + archivos estáticos)
├── README.md             # Documentación e instructivo de ejecución
└── static/               # Frontend desacoplado (Vanilla HTML, CSS, JS)
    ├── index.html        # Landing page semántica y moderna
    ├── style.css         # Diseño visual dark mode, paleta HSL y componentes
    └── app.js            # Lógica de interacción y cliente fetch para inferencia
```

---

## 🚀 Cómo Probar la Aplicación

### Opción 1: En la Nube (Producción Live en Vercel)
Acceso inmediato sin instalar nada:
👉 **[https://desarrollo-de-ia.vercel.app/](https://desarrollo-de-ia.vercel.app/)**

### Opción 2: Localmente en tu Máquina
Desde la raíz del repositorio, ejecutá:

```powershell
py -3.14 demo_app/server.py
```

El servidor iniciará automáticamente en:
👉 **`http://localhost:8000`**


---

## 🧪 Casos de Prueba Incluidos en la Interfaz

La aplicación cuenta con 5 botones de **Prueba Rápida** representativos de los distintos segmentos del catálogo:

1. **Subcompacto Base (13 pulgadas, BRAND_A):**
   - Especificaciones: `155/70 R 13`
   - Costo estimado: **$4.013,05 INR**
2. **Turismo Económico Urbano (14 pulgadas, BRAND_C):**
   - Especificaciones: `175/65 R 14`
   - Costo estimado: **$3.870,80 INR**
3. **Sedan / SUV Confort (17 pulgadas, BRAND_A):**
   - Especificaciones: `225/50 R 17`
   - Costo estimado: **$7.123,11 INR**
4. **Camioneta Alta Gama (20 pulgadas, BRAND_B):**
   - Especificaciones: `275/40 R 20`
   - Costo estimado: **$12.225,91 INR**
5. **Transporte Pesado / Industrial (22.5 pulgadas, BRAND_D):**
   - Especificaciones: `279.4 mm de ancho` (perfil no especificado en catálogo)
   - Costo estimado: **$24.186,33 INR**
   - *Nota técnica:* El pipeline imputa la mediana automáticamente sin arrojar error.

---

## 🔌 Especificación de la API REST

### Endpoint de Predicción
- **Ruta:** `POST /api/predict`
- **Headers:** `Content-Type: application/json`
- **Cuerpo de Solicitud (Ejemplo):**
  ```json
  {
    "rim": 16.0,
    "width": 205.0,
    "ratio": 55.0,
    "brand": "BRAND_A"
  }
  ```
- **Respuesta (Ejemplo):**
  ```json
  {
    "status": "success",
    "input": {
      "rim": 16.0,
      "width": 205.0,
      "ratio": 55.0,
      "brand": "BRAND_A"
    },
    "predicted_cost_inr": 6016.24,
    "formatted_cost": "$6,016.24 INR",
    "model_info": {
      "algorithm": "Random Forest Regressor (200 árboles)",
      "selection_stage": "Ganador 5-Fold Cross-Validation (ISPC Evidencia 3)",
      "cv_rmse_inr": 1381.01,
      "test_r2": 0.8446
    }
  }
  ```

### Endpoint de Estado (Healthcheck)
- **Ruta:** `GET /api/health`
