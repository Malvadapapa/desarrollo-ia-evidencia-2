# Módulo: Evidencia de Aprendizaje N.º 3 (Optimización y Comparación de Modelos)

## 🎯 Objetivo del Módulo

Optimizar sistemáticamente el modelo supervisado para predecir el costo unitario de catálogo (`Unit Cost (INR)`) de neumáticos, empleando **Validación Cruzada (5-Fold CV)** y **Búsqueda Exhaustiva de Hiperparámetros (GridSearchCV)**. Comparar tres enfoques metodológicos (Regresión Lineal OLS baseline, Ridge regularizado y Random Forest), validar la significancia estadística de la mejora sobre el conjunto de prueba independiente ($N=51$) y desplegar la solución en producción mediante serialización con `joblib` e interfaz CLI.

---

## 📚 Conceptos Clave para Estudiar

### 1. Validación Cruzada (K-Fold Cross-Validation)
- **Qué es:** Técnica de remuestreo donde el conjunto de entrenamiento se divide en $K$ pliegues disjuntos (aquí $K=5$). En cada iteración, $K-1$ pliegues entrenan el modelo y el pliegue restante evalúa la generalización.
- **Por qué se usa aquí:** Con 200 observaciones de entrenamiento, una partición fija única es susceptible a sesgos muestrales. La validación cruzada reporta la media y el desvío estándar ($\mu \pm \sigma$) del error, ofreciendo una métrica robusta de estabilidad.
- **Prevención de Data Leakage:** Los estadísticos como la mediana de imputación y la escala deben calcularse exclusivamente sobre los folds de entrenamiento de cada paso.

### 2. Optimización de Hiperparámetros (Grid Search)
- **Qué es:** Procedimiento determinista que explora todas las combinaciones posibles de una grilla de valores predefinidos de hiperparámetros, seleccionando la combinación que minimiza el error de validación cruzada.
- **Diferencia entre Parámetros e Hiperparámetros:** Los parámetros se aprenden analíticamente durante el entrenamiento (ej. los pesos $\beta_j$ de la regresión o las divisiones de los árboles); los hiperparámetros se configuran a priori para guiar el aprendizaje (ej. penalización $\alpha$, profundidad máxima de árboles).

### 3. Regularización Lineal: Ridge ($L_2$)
- **Qué es:** Modificación de la función de costo de Mínimos Cuadrados Ordinarios sumando una penalización sobre la magnitud cuadrática de los coeficientes ($\alpha \sum \beta_j^2$). Contrae coeficientes sin eliminarlos, mitigando la multicolinealidad entre `rim` y `width` ($r = 0,57$).
- **Escalado obligatorio:** Requiere `StandardScaler`, ya que de lo contrario las variables con mayor rango numérico serían penalizadas desproporcionadamente.

### 4. Ensambles de Árboles: Random Forest (Bagging)
- **Qué es:** Construye múltiples árboles independientes (aquí 200) mediante muestreo con reemplazo (*bootstrap*) y subconjuntos aleatorios de variables (*feature bagging*).
- **Por qué supera a los lineales:** El promedio de árboles reduce la varianza drásticamente y es capaz de capturar umbrales discretos y relaciones no lineales complejas (como los sobrecostos estructurales de cubiertas pesadas de `BRAND_D`), donde los hiperplanos lineales fallan.

---

## 📋 Resumen de Resultados Comparativos

| Algoritmo | Familia | CV RMSE (INR) | Test RMSE (INR) | Test MAE (INR) | Test R² | Mejora Test RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión OLS (Ev2)** | Lineal OLS | $2.589,29 \pm 1.350$ | $2.549,50$ | $1.678,60$ | $0,7968$ | Baseline ($0\%$) |
| **Ridge ($\alpha=5.0$)** | Lineal Regularizado ($L_2$) | $2.547,51 \pm 1.257$ | $2.696,96$ | $1.747,88$ | $0,7727$ | $-5,78\%$ |
| **Random Forest (Ganador)** | Ensamble Bagging | **$1.381,01 \pm 532$** | **$2.229,88$** | **$1.243,18$** | **$0,8446$** | **$+12,54\%$** |

---

## 🚀 Puesta en Producción

El modelo ganador se encuentra serializado en:
* [best_model.joblib](file:///c:/Users/av-cr/OneDrive/Escritorio/desarrollo-de-ia/models/best_model.joblib)

Se invoca desde consola ejecutando:
```bash
py src/predict.py --rim 16.0 --width 205.0 --ratio 55.0 --brand BRAND_B
```
O programáticamente:
```python
from src.predict import predict_single_tire

costo = predict_single_tire(rim=16.0, width=205.0, ratio=55.0, brand='BRAND_B')
print(f"Costo: ${costo:,.2f} INR")
```

---

## 📁 Entregables Generados

* **Informe Técnico Markdown:** [EVIDENCIA_3_INFORME.md](file:///c:/Users/av-cr/OneDrive/Escritorio/desarrollo-de-ia/entregables/EVIDENCIA_3_INFORME.md)
* **Documento Académico Word:** [Desarrollo_de_IA_ev_N3.docx](file:///c:/Users/av-cr/OneDrive/Escritorio/desarrollo-de-ia/entregables/Desarrollo_de_IA_ev_N3.docx)
* **Jupyter Notebook Interactivo:** [Evidencia_3_Optimizacion.ipynb](file:///c:/Users/av-cr/OneDrive/Escritorio/desarrollo-de-ia/notebooks/Evidencia_3_Optimizacion.ipynb)
* **Figuras Diagnósticas:** [reports/figures/](file:///c:/Users/av-cr/OneDrive/Escritorio/desarrollo-de-ia/reports/figures/)
