# Desarrollo de Inteligencia Artificial — Proyecto de Modelado Predictivo

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Evaluaci%C3%B3n-Evidencias%201%2C%202%20y%203-brightgreen.svg)]()
[![Vercel Live](https://img.shields.io/badge/Vercel-Producci%C3%B3n%20Live-black?logo=vercel)](https://desarrollo-de-ia.vercel.app/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

> 🚀 **Demostración Interactiva en Producción (Vercel Serverless):**  
> 👉 **[https://desarrollo-de-ia.vercel.app/](https://desarrollo-de-ia.vercel.app/)**

Repositorio oficial del espacio curricular **Desarrollo de Inteligencia Artificial** correspondiente a la **Tecnicatura Superior en Desarrollo de Software** del **Instituto Superior Politécnico Córdoba (ISPC)**.

* **Docente Titular:** Cra. Sol Figueroa
* **Equipo de Desarrollo (TeamYear3):**
  * **Vargas, Cristian**
  * **Quinteros, Karina**
  * **Zárate, Laura**
* **Ciclo Lectivo:** 2026

---

## 📋 Resumen del Proyecto

El objetivo central de este proyecto es el diseño, entrenamiento, optimización y puesta en producción de modelos de aprendizaje automático supervisado para predecir el **costo unitario de catálogo (`Unit Cost (INR)`)** de neumáticos a partir de sus dimensiones físicas y marca fabricante.

A lo largo del trayecto pedagógico se articularon tres etapas integradas:
1. **Evidencia N.º 1 (Ingeniería de Características y Extracción):** Análisis exploratorio de datos (EDA), depuración del catálogo estático de inventario e ingeniería de variables mediante expresiones regulares para extraer cotas geométricas (`rim`, `width`, `ratio`) a partir de nomenclaturas industriales en el campo `Product Name`.
2. **Evidencia N.º 2 (Modelado Predictivo Base):** Formulación de una línea de base mediante Regresión Lineal Múltiple por Mínimos Cuadrados Ordinarios (OLS), validando supuestos econométricos e identificando la presencia de colinealidad geométrica y heterocedasticidad en el segmento pesado.
3. **Evidencia N.º 3 (Optimización y Comparación de Algoritmos):** Rediseño metodológico completo con pipelines libres de fuga de datos (*data leakage*), optimización de hiperparámetros mediante **GridSearchCV con Validación Cruzada de 5 Pliegues (5-Fold CV)**, comparación rigurosa entre tres familias algorítmicas (Lineal OLS, Regularizado Ridge y Ensamble Random Forest) y **puesta en producción** con serialización de artefactos, CLI de inferencia y aplicación web interactiva.

---

## 📂 Estructura del Repositorio

El repositorio conserva únicamente el código fuente ejecutable, datos limpios, notebooks renderizadas y componentes operativos:

```
Desarrollo-de-IA/
├── data/                                 # Datasets limpios y muestras de inferencia
│   ├── DS_Data_Set_Current_Stock.csv    # Catálogo de inventario base (N=251 válidos)
│   ├── DS_Data_Set_Sales.csv            # Registro transaccional de ventas
│   ├── DS_Data_Set_Supplier.csv         # Registro maestro de proveedores
│   ├── batch_sample.csv                 # Muestra para validación batch en producción
│   └── batch_predictions.csv            # Salida generada por la inferencia batch
│
├── demo_app/                             # Aplicación web desacoplada de producción
│   ├── server.py                        # Servidor HTTP liviano (REST API /api/predict)
│   ├── README.md                        # Guía de despliegue y uso de la demo web
│   └── static/                          # Frontend interactivo (HTML5, Vanilla CSS, JS)
│       ├── index.html                   # Landing page moderna y responsiva
│       ├── style.css                    # Sistema de diseño dark mode con paleta HSL
│       └── app.js                       # Cliente asíncrono para inferencia en tiempo real
│
├── models/                               # Artefactos de modelos serializados
│   └── best_model.joblib                # Pipeline autónomo entrenado (preprocesador + RF)
│
├── notebooks/                            # Cuadernos interactivos ejecutados y documentados
│   ├── Evidencia_2_Modelado_Regresion.ipynb  # Modelado y diagnóstico de Regresión Lineal
│   └── Evidencia_3_Optimizacion.ipynb        # GridSearch, CV 5-Fold, comparación y gráficos
│
├── reports/figures/                      # Figuras diagnósticas generadas en alta resolución
│   ├── diagnostico_regresion.png        # Diagnóstico residual OLS (Evidencia 2)
│   ├── ev3_cv_comparacion_modelos.png   # Boxplot comparativo en Validación Cruzada
│   ├── ev3_importancia_variables.png    # Ranking MDI de importancia de variables
│   ├── ev3_reales_vs_predichos.png      # Gráfico de dispersión frente a identidad y=x
│   └── ev3_diagnostico_residuos.png     # Dispersión y densidad residual de Random Forest
│
├── src/                                  # Módulos Python reutilizables
│   ├── __init__.py                      # Inicializador de paquete
│   ├── data_preprocessing.py            # Extracción Regex y limpieza de datos (Ev1/Ev2)
│   ├── regression_models.py             # Entrenamiento y métricas de regresión (Ev2)
│   ├── model_optimization.py            # Pipelines, GridSearchCV y comparación pareada (Ev3)
│   └── predict.py                       # CLI de inferencia en producción (individual y batch)
│
├── tests/                                # Suite de pruebas automatizadas
│   ├── test_pipeline.py                 # Pruebas de extracción y preparación de datos
│   └── test_optimization.py             # 20 pruebas de no-fuga, optimización y reproducibilidad
│
├── .gitignore                            # Exclusiones de Git
├── requirements.txt                      # Dependencias del entorno
└── README.md                             # Documentación principal del proyecto
```

---

## ⚙️ Metodología y Prevención de Fuga de Datos (*Data Leakage*)

Para asegurar una estimación imparcial de la capacidad de generalización en un conjunto acotado ($N = 251$ observaciones válidas, divididas en $N = 200$ para Entrenamiento y $N = 51$ para Prueba independiente con `random_state=42`), todas las transformaciones se encapsularon dentro de un objeto `ColumnTransformer` integrado a cada `Pipeline`:

* **Imputación de Perfil (`ratio`):** Se aplica `SimpleImputer(strategy='median')`. La mediana se calcula estrictamente sobre el bloque de entrenamiento de cada pliegue de la validación cruzada, completando de forma transparente los neumáticos comerciales que carecen de especificación de perfil.
* **Codificación de Marca (`Brand`):** `OneHotEncoder(categories=[['BRAND_A', 'BRAND_B', 'BRAND_C', 'BRAND_D']], drop=['BRAND_A'])`. Se mantiene `BRAND_A` como categoría base de referencia, eliminando el riesgo de multicolinealidad perfecta y garantizando consistencia categórica ante cualquier pliegue.
* **Escalado Condicional (`StandardScaler`):** Incorporado únicamente en la Regresión Ridge para que la penalización $L_2$ pondere equitativamente variables con escalas disímiles (milímetros, pulgadas y porcentajes). Se omite en Random Forest debido a la invariancia de escala de los árboles de decisión.

---

## 📊 Comparación Sistemática de Algoritmos

Se evaluaron tres paradigmas de aprendizaje supervisado bajo **Validación Cruzada de 5 Pliegues** sobre el conjunto de entrenamiento ($N = 200$):

1. **Regresión Lineal OLS (Baseline Evidencia 2):** Modelo lineal aditivo paramétrico de referencia.
2. **Regresión Ridge ($L_2$):** Búsqueda sistemática del hiperparámetro de penalización $\alpha \in [0.01, 500.0]$. La configuración óptima resultó $\alpha = 5.0$.
3. **Random Forest Regressor (Ensamble por Bagging):** Exploración de grilla exhaustiva (48 combinaciones) sobre `n_estimators`, `max_depth`, `min_samples_leaf` y `max_features`. Configuración óptima: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`.

### Tabla Consolidada de Desempeño

| Algoritmo | Configuración Óptima | CV RMSE (INR) ↓ | CV MAE (INR) ↓ | CV $R^2$ ↑ | Test RMSE (INR) ↓ | Test MAE (INR) ↓ | Test $R^2$ ↑ | Veredicto |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS Baseline** | Analítico directo | $2.589,29 \pm 1.350,21$ | $1.482,50 \pm 266,81$ | 0,7576 | $2.549,50$ | $1.678,60$ | 0,7968 | *Línea Base* |
| **Ridge ($L_2$)** | $\alpha = 5.0$ | $2.547,51 \pm 1.257,18$ | $1.535,53 \pm 320,55$ | 0,7698 | $2.696,96$ | $1.747,88$ | 0,7727 | *Sin mejora en Test* |
| **Random Forest** | **200 árboles, sqrt** | **$1.381,01 \pm 532,77$** | **$805,46 \pm 202,17$** | **0,9280** | **$2.229,88$** | **$1.243,18$** | **0,8446** | **🏆 Ganador Indiscutido** |

### Análisis Pareado Pliegue por Pliegue (5-Fold CV)

Para verificar si la mejora era consistente o producto de un pliegue aislado, se contrastó el error cuadrático pliegue a pliegue:

| Pliegue | Regresión OLS (Baseline) | Regresión Ridge ($\alpha=5.0$) | Random Forest Regressor | Reducción de Error (OLS vs RF) |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | $2.023,4\text{ INR}$ | $2.011,2\text{ INR}$ | **$1.212,4\text{ INR}$** | **$+811,0\text{ INR}$** |
| **Fold 2** | $2.212,0\text{ INR}$ | $2.112,0\text{ INR}$ | **$2.187,1\text{ INR}$** | **$+24,9\text{ INR}$** |
| **Fold 3** | $1.881,8\text{ INR}$ | $2.016,4\text{ INR}$ | **$880,1\text{ INR}$** | **$+1.001,7\text{ INR}$** |
| **Fold 4** | $5.257,2\text{ INR}$ | $5.033,0\text{ INR}$ | **$1.803,1\text{ INR}$** | **$+3.454,1\text{ INR}$** |
| **Fold 5** | $1.572,1\text{ INR}$ | $1.564,9\text{ INR}$ | **$822,4\text{ INR}$** | **$+749,7\text{ INR}$** |
| **Promedio** | **$2.589,3 \pm 1.350,2$** | **$2.547,5 \pm 1.257,2$** | **$1.381,0 \pm 532,8$** | **$+1.208,3 \pm 1.309,1\text{ INR}$** |

> **Conclusión Metodológica:** Random Forest superó a la regresión lineal en el **100% de los pliegues (5 de 5)**. En el Fold 4 —donde recayeron varios neumáticos pesados que dispararon el error lineal por encima de los 5.200 INR—, Random Forest contuvo el RMSE en 1.803 INR, demostrando su capacidad para asimilar las no linealidades del segmento industrial.

---

## 🔍 Interpretabilidad e Importancia de Variables

A partir de la reducción media de impureza (*Mean Decrease in Impurity*, MDI) en los 200 árboles del ensamble:

* **Ancho de Sección (`width`):** **$40,3\%$**
* **Diámetro del Rin (`rim`):** **$32,1\%$**
* **Marca Industrial Pesada (`BRAND_D`):** **$13,2\%$**
* **Relación de Aspecto (`ratio`):** **$11,9\%$**
* **Marcas Estándar (`BRAND_C` y `BRAND_B`):** **$2,5\%$ acumulado**

Las variables geométricas explican conjuntamente más del **$72\%$** del costo unitario, guardando estricta correspondencia física con el volumen de material e insumos requeridos para la fabricación de cada cubierta.

---

## 🚀 Puesta en Producción

El pipeline ganador fue exportado a disco en [`models/best_model.joblib`](models/best_model.joblib). El artefacto es completamente autónomo y no requiere ingeniería de variables manual externa.

### 1. Despliegue en la Nube (Vercel Serverless) y Demo Local
* 🌐 **Producción Live (Vercel):** 👉 **[https://desarrollo-de-ia.vercel.app/](https://desarrollo-de-ia.vercel.app/)**
* 💻 **Ejecución local desacoplada:**
  ```powershell
  py -3.14 demo_app/server.py
  ```
  Abrir en el navegador local: 👉 **`http://localhost:8000`**

### 2. Inferencia Individual por Consola (CLI)
Mediante el script [`src/predict.py`](src/predict.py):

```powershell
# Caso 1: Turismo económico 14" (BRAND_C)
py -3.14 src/predict.py --rim 14.0 --width 175.0 --ratio 65.0 --brand BRAND_C
# -> Costo Estimado: $3,870.80 INR

# Caso 2: Turismo estándar 17" (BRAND_A)
py -3.14 src/predict.py --rim 17.0 --width 225.0 --ratio 50.0 --brand BRAND_A
# -> Costo Estimado: $7,123.11 INR

# Caso 3: Transporte pesado 22.5" sin perfil explícito (BRAND_D)
py -3.14 src/predict.py --rim 22.5 --width 279.4 --brand BRAND_D
# -> Costo Estimado: $24,186.33 INR (imputación automática de mediana)
```

### 3. Inferencia Masiva por Lotes (Batch)
Para procesar inventarios o catálogos completos en formato CSV:

```powershell
py -3.14 src/predict.py --input-csv data/batch_sample.csv --output-csv data/batch_predictions.csv
```

### 4. Consumo Programático en Python (API / Backend)

```python
import joblib
import pandas as pd

# Cargar el pipeline completo (preprocesador + modelo)
pipeline = joblib.load("models/best_model.joblib")

# Entrada en crudo
datos = pd.DataFrame([{
    "rim": 16.0,
    "width": 205.0,
    "ratio": 55.0,
    "Brand": "BRAND_A"
}])

# Predicción directa
costo_inr = pipeline.predict(datos)[0]
print(f"Costo estimado: ${costo_inr:,.2f} INR")
```

---

## 🧪 Pruebas Automatizadas y Reproducibilidad

El proyecto cuenta con una suite de pruebas unitarias que valida:
* Aislamiento de transformaciones sin fuga de datos.
* Coincidencia de categorías en OneHotEncoder.
* Reproducibilidad numérica de la optimización con semilla fija (`random_state=42`).
* Consistencia del modelo serializado en inferencia.

Para ejecutar la suite completa de 20 pruebas:

```powershell
py -3.14 -m unittest discover tests/
```

Salida esperada:
```
....................
----------------------------------------------------------------------
Ran 20 tests in ~17s

OK
```

---

## 📦 Instalación y Configuración del Entorno

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/TeamYear3/Desarrollo-de-IA.git
   cd Desarrollo-de-IA
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux / macOS:
   source .venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Explorar las Notebooks:**
   ```bash
   jupyter notebook notebooks/Evidencia_3_Optimizacion.ipynb
   ```

---

## 🎓 Créditos y Marco Institucional

Proyecto desarrollado en el marco de la **Tecnicatura Superior en Desarrollo de Software** del **ISPC (Instituto Superior Politécnico Córdoba)** para la cátedra de **Desarrollo de Inteligencia Artificial**, dictada por la **Cra. Sol Figueroa** (Ciclo Lectivo 2026).
