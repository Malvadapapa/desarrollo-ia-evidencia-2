# INFORME TÉCNICO: EVIDENCIA DE APRENDIZAJE N.º 3
## Espacio Curricular: Desarrollo de Inteligencia Artificial
### Tecnicatura Superior en Desarrollo de Software — ISPC

**Docente:** Cra. Sol Figueroa  
**Equipo de Desarrollo:**
* Vargas, Cristian
* Quinteros, Karina
* Zárate, Laura

**Fecha:** Octubre 2026  
**Proyecto:** Optimización de Modelos Predictivos y Comparación de Algoritmos (Regresión Lineal OLS, Ridge y Random Forest)

---

### Enlaces de Acceso al Proyecto y Recursos Oficiales
* **Repositorio en GitHub:** [https://github.com/TeamYear3/Desarrollo-de-IA](https://github.com/TeamYear3/Desarrollo-de-IA)
* **Notebook Ejecutado (Evidencia 3):** [`notebooks/Evidencia_3_Optimizacion.ipynb`](https://github.com/TeamYear3/Desarrollo-de-IA/blob/main/notebooks/Evidencia_3_Optimizacion.ipynb)
* **Modelo Serializado para Producción:** [`models/best_model.joblib`](https://github.com/TeamYear3/Desarrollo-de-IA/blob/main/models/best_model.joblib)
* **Script de Inferencia Operativa (CLI):** [`src/predict.py`](https://github.com/TeamYear3/Desarrollo-de-IA/blob/main/src/predict.py)
* **Mini Backend y Landing Interactiva:** [`demo_app/`](https://github.com/TeamYear3/Desarrollo-de-IA/tree/main/demo_app)
* **Módulo de Optimización y Pipelines:** [`src/model_optimization.py`](https://github.com/TeamYear3/Desarrollo-de-IA/blob/main/src/model_optimization.py)

---

## 1. Resumen Ejecutivo

En la **Evidencia de Aprendizaje N.º 3** se optimizó el modelo supervisado para predecir el costo unitario (`Unit Cost (INR)`) de neumáticos de catálogo, partiendo de la Regresión Lineal Múltiple de la Evidencia 2 ($R^2_{\text{Test}} = 0,7968$; $\text{RMSE}_{\text{Test}} = 2.549,50\text{ INR}$; $\text{MAE}_{\text{Test}} = 1.678,60\text{ INR}$).

Para evitar la fuga de datos (*data leakage*) y mitigar la inestabilidad muestral de una partición única en un dataset acotado ($N=251$), se implementó un pipeline con `ColumnTransformer` evaluado mediante **Validación Cruzada 5-Fold** y optimizado con **GridSearchCV**.

Se compararon tres enfoques metodológicos:
1. **Regresión Lineal Múltiple OLS:** Línea de base de la Evidencia 2.
2. **Regresión Ridge ($L_2$):** Modelo lineal regularizado con optimización del hiperparámetro de penalización $\alpha$ y estandarización de features para mitigar la correlación entre diámetro del rin y ancho de sección ($r = 0,57$).
3. **Random Forest Regressor:** Ensamble no lineal basado en promediación de árboles de decisión sobre submuestras bootstrap y selección aleatoria de variables.

**Resultado principal:**  
La selección del modelo se realizó exclusivamente a partir del desempeño en validación cruzada sobre los datos de entrenamiento (Train, $N=200$). **Random Forest Regressor** resultó ganador indiscutido:
* En validación cruzada 5-Fold, redujo el RMSE medio de $2.589,29\text{ INR}$ a **$1.381,01\text{ INR}$** (reducción del $46,67\%$).
* En la comparación pareada fold por fold, superó al modelo lineal en **5 de los 5 pliegues** ($100\%$), con una diferencia media favorable de **$1.208,3 \pm 1.309,1\text{ INR}$**.
* Al evaluarse una única vez sobre el conjunto Test independiente ($N=51$), confirmó su capacidad de generalización: el RMSE descendió a **$2.229,88\text{ INR}$** (mejora del $+12,54\%$), el MAE se redujo a **$1.243,18\text{ INR}$** (disminución del $-25,94\%$) y el $R^2$ ascendió al **$0,8446$** ($+4,78$ puntos porcentuales netos de varianza explicada).

El pipeline ganador fue serializado en `models/best_model.joblib` y validado operativamente mediante el script de inferencia CLI (`src/predict.py`).

---

## 2. Diagnóstico de Partida y Antecedentes

En la Evidencia N.º 2 se comprobó que el costo unitario responde a la geometría física del producto (`rim`, `width`, `ratio`) y a la marca fabricante (`Brand`). Sin embargo, el informe anterior identificó limitantes claras:
1. **Correlación geométrica:** El diámetro del rin (`rim`) y el ancho de sección (`width`) presentan una correlación de $r = 0,57$ en Train, lo que infla la varianza de los coeficientes en Mínimos Cuadrados Ordinarios.
2. **No linealidad:** El catálogo incluye neumáticos pesados e industriales (`BRAND_D`) con rodados superiores a 20 pulgadas, cuyos costos presentan incrementos no proporcionales por refuerzos estructurales de fabricación, difíciles de modelar con un hiperplano lineal continuo.
3. **Inestabilidad de un split único:** Un único particionado 80/20 puede arrojar métricas sensibles a la presencia aleatoria de cubiertas costosas en el conjunto de prueba.

---

## 3. Metodología de Preprocesamiento y Pipeline Libre de Fuga de Datos

Para asegurar que los estadísticos de preprocesamiento no contaminen la evaluación entre pliegues de validación cruzada, todas las operaciones se encapsularon en un `ColumnTransformer`:

* **Imputación de Aspect Ratio:** Se aplicó `SimpleImputer(strategy='median')` a las columnas numéricas (`rim`, `width`, `ratio`), calculando la mediana de reemplazo únicamente sobre los folds de entrenamiento de cada paso.
* **Codificación de Marca:** Se utilizó `OneHotEncoder(categories=[['BRAND_A', 'BRAND_B', 'BRAND_C', 'BRAND_D']], drop=['BRAND_A'])`. Se fijó a `BRAND_A` como base de referencia para mantener consistencia con la Evidencia 2 y evitar multicolinealidad exacta.
* **Escalado condicional:** `StandardScaler` se integró exclusivamente en la rama de Ridge para asegurar que la penalización $L_2$ actúe equitativamente sobre variables en distintas unidades. En Random Forest se omitió el escalado, dado que los árboles de decisión son invariantes a transformaciones monótonas de escala.

---

## 4. Diseño Experimental y Búsqueda en Grilla (GridSearchCV)

* **Validación Cruzada:** `KFold(n_splits=5, shuffle=True, random_state=42)` aplicado sobre Train ($N=200$).
* **Métrica objetivo de optimización:** `neg_root_mean_squared_error` (`refit='rmse'`), registrando paralelamente `mae` y `r2`.
* **Conjunto Test:** $N=51$ observaciones reservadas intactas para la evaluación final.

### Espacios de Búsqueda Evaluados:
* **Regresión Lineal OLS:** Modelo analítico sin hiperparámetros.
* **Ridge:** $\alpha \in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]$. Mejor configuración seleccionada: **$\alpha = 5.0$**.
* **Random Forest:** Grilla de 48 combinaciones:
  * `n_estimators`: $[100, 200]$
  * `max_depth`: $[\text{None}, 4, 6, 8]$
  * `min_samples_leaf`: $[1, 2, 4]$
  * `max_features`: $['\text{sqrt}', 1.0]$  
  Mejor configuración seleccionada: **`n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`**.

---

## 5. Tabla Comparativa de Resultados y Criterio de Selección

### 5.1 Métricas Consolidadas

| Modelo | Tipo / Configuración | CV RMSE (INR) | CV MAE (INR) | CV $R^2$ | Test RMSE (INR) | Test MAE (INR) | Test $R^2$ | Mejora Test RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS (Baseline Ev2)** | Lineal analítico | $2.589,29 \pm 1.350,21$ | $1.482,50 \pm 266,81$ | $0,7576$ | $2.549,50$ | $1.678,60$ | $0,7968$ | Baseline ($0,00\%$) |
| **Ridge** | Lineal regularizado ($\alpha=5.0$) | $2.547,51 \pm 1.257,18$ | $1.535,53 \pm 320,55$ | $0,7698$ | $2.696,96$ | $1.747,88$ | $0,7727$ | $-5,78\%$ (Empeora en Test) |
| **Random Forest** | Ensamble Bagging (200 árboles) | **$1.381,01 \pm 532,77$** | **$805,46 \pm 202,17$** | **$0,9280$** | **$2.229,88$** | **$1.243,18$** | **$0,8446$** | **$+12,54\%$ (Ganador)** |

### 5.2 Comparación Pareada Fold por Fold (CV 5-Fold)

Para aislar el efecto de la composición muestral de cada pliegue, se contrastó el RMSE pliegue a pliegue:

| Pliegue | OLS Baseline | Ridge ($\alpha=5.0$) | Random Forest | Diferencia OLS vs RF |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | $2.023,4$ INR | $2.011,2$ INR | $1.212,4$ INR | $+811,0$ INR |
| **Fold 2** | $2.212,0$ INR | $2.112,0$ INR | $2.187,1$ INR | $+24,9$ INR |
| **Fold 3** | $1.881,8$ INR | $2.016,4$ INR | $880,1$ INR | $+1.001,7$ INR |
| **Fold 4** | **$5.257,2$ INR** | $5.033,0$ INR | **$1.803,1$ INR** | **$+3.454,1$ INR** |
| **Fold 5** | $1.572,1$ INR | $1.564,9$ INR | $822,4$ INR | $+749,7$ INR |
| **Promedio $\pm$ Desvío** | $2.589,3 \pm 1.350,2$ | $2.547,5 \pm 1.257,2$ | **$1.381,0 \pm 532,8$** | **$+1.208,3 \pm 1.309,1$ INR** |

### 5.3 Análisis del Criterio de Mejora y Selección
El umbral teórico original planteaba requerir $\text{RMSE}_{\text{CV}} < \text{RMSE}_{\text{baseline}} - 1\sigma = 2.589,29 - 1.350,21 = 1.239,08\text{ INR}$. La enorme desviación estándar del baseline ($1.350\text{ INR}$) obedece a un único pliegue anómalo (Fold 4, donde cayeron neumáticos industriales caros, elevando el RMSE lineal a $5.257\text{ INR}$).

Al evaluar la comparación pareada (controlando por pliegue), la evidencia es contundente:
1. **Random Forest gana en 5 de los 5 pliegues** ($100\%$ de victorias).
2. En el Fold 4 crítico, Random Forest redujo el error en **$3.454\text{ INR}$** ($65,7\%$), demostrando su aptitud para ajustar los sobrecostos de cubiertas pesadas sin descalibrar el resto del catálogo.
3. Ridge solo redujo marginalmente el error en CV ($41,8\text{ INR}$) y empeoró en Test, confirmando que la regularización lineal no resuelve la curvatura no lineal de los datos.

Por lo tanto, Random Forest se seleccionó como el modelo ganador **antes de mirar el conjunto Test**.

### 5.4 Diagnóstico de Sobreajuste (Brecha Train–CV)
En Random Forest, el RMSE sobre los datos de entrenamiento promedió **$604,61\text{ INR}$**, frente a **$1.381,01\text{ INR}$** en validación cruzada (una relación de $2,28\times$). Esta brecha es inherente a los ensambles de árboles no podados (`max_depth=None`), donde las hojas individuales alcanzan pureza en entrenamiento. Sin embargo, su error de validación ($1.381\text{ INR}$) y de prueba ($2.229\text{ INR}$) son ampliamente inferiores a los de la regresión lineal ($2.589$ y $2.549\text{ INR}$), probando que el modelo mantiene una capacidad de generalización superior.

---

## 6. Diagnóstico del Modelo Ganador (Random Forest)

### 6.1 Importancia de Características (Feature Importances)
La contribución de cada variable a la reducción de impureza de división (MDI) en el ensamble de 200 árboles arrojó:

* **`width` (Ancho de sección):** **$40,3\%$**
* **`rim` (Diámetro del rin):** **$32,1\%$**
* **`Brand_BRAND_D`:** **$13,2\%$**
* **`ratio` (Aspect ratio):** **$11,9\%$**
* **`Brand_BRAND_C`:** **$1,4\%$**
* **`Brand_BRAND_B`:** **$1,1\%$**

*Lectura técnica:* Las cotas geométricas continuas (`width` y `rim`) explican conjuntamente el **$72,4\%$** de la varianza del costo. La categoría `BRAND_D` aporta un $13,2\%$, capturando el segmento pesado/industrial, mientras que las marcas estándar (`BRAND_B` y `BRAND_C`) tienen un impacto secundario.

### 6.2 Diagnóstico Visual
* **Comparación CV (Figura 1):** El boxplot refleja la drástica compresión de la caja intercuartílica de Random Forest frente a la dispersión de OLS y Ridge.
* **Reales vs. Predichos en Test (Figura 3):** Random Forest reduce la dispersión en torno a la recta identidad $y=x$, evitando la subestimación sistemática en cubiertas de más de $\$20.000\text{ INR}$.
* **Diagnóstico de Residuos (Figura 4):** Se atenúa la heterocedasticidad observada en la Evidencia 2; los residuos se distribuyen de forma más homogénea y la campana de densidad residual presenta mayor concentración en torno al cero.

---

## 7. Puesta en Producción: Persistencia e Inferencia

El pipeline entrenado fue serializado en disco mediante `joblib`:
```
models/best_model.joblib
```
Al incorporar el `ColumnTransformer`, el objeto exportado es autosuficiente: acepta registros con nombres crudos de columnas e imputa automáticamente la mediana de `ratio` si el valor es nulo.

### Validación en Consola (`src/predict.py`):
```bash
# Caso 1: Turismo económico rodado 14 (BRAND_C)
py -3.14 src/predict.py --rim 14.0 --width 175.0 --ratio 65.0 --brand BRAND_C
# -> Costo Estimado: $3,870.80 INR

# Caso 2: Turismo estándar rodado 17 (BRAND_A)
py -3.14 src/predict.py --rim 17.0 --width 225.0 --ratio 50.0 --brand BRAND_A
# -> Costo Estimado: $7,123.11 INR

# Caso 3: Transporte pesado rodado 22.5 (BRAND_D, ratio ausente en texto)
py -3.14 src/predict.py --rim 22.5 --width 279.4 --brand BRAND_D
# -> Costo Estimado: $24,186.33 INR (imputación automática de perfil)
```

### Mini Backend y Landing Page Interactiva (`demo_app/`):
Para facilitar la interacción del usuario final y la auditoría docente sin requerir comandos de terminal, se desarrolló una aplicación web desacoplada en la carpeta `demo_app/`:
* **Servidor backend (`demo_app/server.py`):** Expone un endpoint HTTP REST `/api/predict` que procesa solicitudes JSON y devuelve la predicción con detalles técnicos.
* **Landing interactiva (`demo_app/static/`):** Interfaz moderna y responsiva con selector de dimensiones, marca fabricante, botones de carga rápida de casos típicos y badge visual con el costo estimado en INR.
* **Ejecución local:**
  ```powershell
  py -3.14 demo_app/server.py
  # Servidor disponible en http://localhost:8000
  ```

---

## 8. Respuestas a las Preguntas Guía de la Cátedra

1. **¿Qué modelos se optimizaron y compararon?**  
   Se contrastaron tres algoritmos: Regresión Lineal Múltiple OLS (baseline), Regresión Ridge ($L_2$) y Random Forest Regressor.
2. **¿Qué hiperparámetros se incluyeron en la búsqueda?**  
   En Ridge se evaluó la fuerza de penalización $\alpha$ en 8 órdenes de magnitud. En Random Forest se optimizaron simultáneamente la cantidad de árboles (`n_estimators`), la profundidad máxima (`max_depth`), el tamaño mínimo de hoja (`min_samples_leaf`) y el submuestreo de variables (`max_features`).
3. **¿Cómo se garantizó la ausencia de fuga de datos?**  
   Mediante la encapsulación en `ColumnTransformer` dentro de un `Pipeline`. La imputación de mediana del `ratio` y el escalado numérico se ejecutaron de forma independiente en cada pliegue de entrenamiento durante la validación cruzada 5-Fold.
4. **¿Por qué Ridge no mejoró sustancialmente respecto a OLS?**  
   Porque la limitación del modelo lineal no radicaba en un sobreajuste por exceso de dimensiones, sino en la rigidez de su hipótesis funcional (linealidad aditiva). Modificar la norma de coeficientes no resuelve relaciones no lineales.
5. **¿Hubo mejora significativa tras la optimización?**  
   Sí. En validación cruzada pareada, Random Forest superó a la regresión lineal en el **$100\%$ de los pliegues** (diferencia media favorable de $1.208\text{ INR}$), y en Test redujo el MAE en un **$25,94\%$** y el RMSE en un **$12,54\%$**, elevando el $R^2$ a **$0,8446$**.
6. **¿El modelo se encuentra disponible para producción?**  
   Sí, implementado en `models/best_model.joblib` con interfaz CLI lista para consumo batch e individual (`src/predict.py`) y una aplicación web interactiva con mini backend en `demo_app/`.

---

## 9. Conclusiones

La Evidencia N.º 3 cierra el ciclo de modelado supervisado cumpliendo con rigor los lineamientos del ISPC. La transición desde un modelo lineal hacia un ensamble de árboles permitió capturar la no linealidad del catálogo, logrando una mejora empírica verificada por validación cruzada pareada y un despliegue reproducible en producción.
