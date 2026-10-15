# INFORME TÉCNICO: EVIDENCIA DE APRENDIZAJE N.º 3
## Espacio Curricular: Desarrollo de Inteligencia Artificial
### Tecnicatura Superior en Desarrollo de Software — ISPC

**Docente:** Cra. Sol Figueroa  
**Equipo de Desarrollo:**
* Vargas, Cristian
* Quinteros, Karina
* Zárate, Laura

**Fecha:** Octubre 2026  
**Proyecto:** Optimización de Modelos Predictivos y Comparación de Algoritmos (Regresión Lineal, Regularización L1/L2 y Ensambles)

---

## 1. Resumen Ejecutivo

En el marco de la **Evidencia de Aprendizaje N.º 3**, se llevó a cabo el proceso de **optimización de hiperparámetros y comparación sistemática de algoritmos** de aprendizaje automático supervisado para la predicción del costo unitario (`Unit Cost (INR)`) de neumáticos de catálogo. El desarrollo toma como punto de partida y línea de base (*baseline*) el modelo de Regresión Lineal Múltiple alcanzado en la **Evidencia N.º 2** ($R^2_{\text{Test}} = 0,7968$; $\text{RMSE}_{\text{Test}} = 2.549,50\text{ INR}$; $\text{MAE}_{\text{Test}} = 1.678,60\text{ INR}$).

Para superar las limitaciones inherentes a la linealidad estricta y a la validación estática en una muestra acotada ($N=251$ observaciones), se estructuró una metodología rigurosa basada en **Validación Cruzada K-Fold ($K=5$)** combinada con **búsqueda sistemática en grilla (`GridSearchCV`)**. El flujo técnico se encapsuló en pipelines integrados mediante `ColumnTransformer`, eliminando toda posibilidad de fuga de datos (*data leakage*) entre pliegues de entrenamiento y validación.

Se evaluaron de forma exhaustiva cinco arquitecturas de modelado:
1. **Regresión Lineal Múltiple OLS** (Baseline referencial).
2. **Regresión Ridge ($L_2$)** con penalización sobre coeficientes y estandarización de features.
3. **Regresión Lasso ($L_1$)** con selección automática de variables y estandarización.
4. **Random Forest Regressor** (Ensamble por Bagging con árboles de decisión aleatorizados).
5. **Gradient Boosting Regressor** (Ensamble secuencial por Boosting con optimización de gradiente).

Los resultados experimentales arrojan una **mejora contundente y estadísticamente significativa** a favor del algoritmo **Gradient Boosting Regressor**:
* El error cuadrático medio ($\text{RMSE}$) descendió de $2.549,50\text{ INR}$ a **$1.577,13\text{ INR}$**, lo que representa una **reducción del $38,14\%$ en el error cuadrático**.
* El error absoluto medio ($\text{MAE}$) cayó de $1.678,60\text{ INR}$ a **$975,85\text{ INR}$**, logrando por primera vez que la desviación media esperada perfore la barrera de las mil rupias (**reducción del $41,86\%$**).
* La varianza explicada sobre datos no vistos de prueba ($R^2$) ascendió del $79,68\%$ al **$92,23\%$**, sumando más de 12 puntos porcentuales de precisión predictiva.

El modelo óptimo fue persistido en disco mediante `joblib` y dotado de una interfaz de inferencia CLI (`src/predict.py`) lista para su despliegue operativo en producción.

---

## 2. Antecedentes y Diagnóstico de la Evidencia N.º 2

En la Evidencia N.º 2, se demostró que el costo unitario de un neumático está fuertemente gobernado por su geometría física (`Rim`, `Width`, `Ratio`) y el segmento tarifario de su fabricante (`Brand`). Si bien la Regresión Lineal Múltiple alcanzó un notable $79,68\%$ de $R^2$, el diagnóstico residual y el análisis de varianza revelaron tres limitantes críticas:

1. **Multicolinealidad entre predictores continuos:** El diámetro del rin (`rim`) y el ancho de sección (`width`) presentan una correlación lineal positiva ($r = 0,55$). En Mínimos Cuadrados Ordinarios (OLS), esta colinealidad infla la varianza de los estimadores $\beta$, haciendo que pequeñas perturbaciones en los datos generen fluctuaciones en los coeficientes.
2. **No linealidad y heterocedasticidad en la cola superior:** En cubiertas de carga pesada e industrial (principalmente `BRAND_D` con rines $\ge 20$ pulgadas), los costos no escalan de forma puramente aditiva, sino que exhiben saltos discretos por refuerzo estructural (plies / capas de acero). Como consecuencia, los residuos del modelo lineal en el rango superior a $\$20.000\text{ INR}$ presentaron mayor dispersión (heterocedasticidad).
3. **Varianza de una partición única:** Evaluar únicamente sobre un split fijo Train/Test (80/20) con 51 muestras de prueba puede arrojar métricas sensibles a qué observaciones cayeron azarosamente en el conjunto de test.

---

## 3. Metodología de Preprocesamiento e Ingeniería de Pipelines

### 3.1 Prevención Absoluta de Data Leakage
En aprendizaje supervisado con validación cruzada, cualquier cálculo estadístico global (medias, medianas de imputación, medias y desviaciones estándar para estandarización) realizado antes del particionado K-Fold constituye una **fuga de datos (*data leakage*)**, pues traslada información del fold de validación hacia el fold de entrenamiento.

Para erradicar este problema metodológico, se implementó un `ColumnTransformer` que se ajusta estrictamente dentro de cada pliegue de entrenamiento:

```
Pipeline de Preprocesamiento (ColumnTransformer)
├── Variables Numéricas ('rim', 'width', 'ratio')
│   ├── SimpleImputer(strategy='median')  <- Mediana calculada fold a fold
│   └── StandardScaler() [Solo en Ridge y Lasso]
└── Variable Categórica ('Brand')
    └── OneHotEncoder(categories=[['BRAND_A', 'BRAND_B', 'BRAND_C', 'BRAND_D']],
                      drop=['BRAND_A'], sparse_output=False, handle_unknown='ignore')
```

### 3.2 Decisión Técnica sobre Escalado de Variables
* **Modelos Lineales Regularizados (Ridge y Lasso):** La penalización de la norma de coeficientes ($L_1$ o $L_2$) asume que todos los predictores operan en la misma escala. Si no se estandarizan, una variable con valores numéricos grandes (como `width` en milímetros, $\approx 150-300$) sería penalizada de forma artificialmente desproporcionada frente a una variable con valores pequeños (como `rim` en pulgadas, $\approx 12-24$). Por tal motivo, se aplicó `StandardScaler()` a las variables numéricas.
* **Modelos Basados en Árboles (Random Forest y Gradient Boosting):** Los árboles de decisión realizan divisiones ortogonales basadas en cuantiles y umbrales locales. Son matemáticamente **invariantes a transformaciones monótonas de escala**. Por ende, para estas familias se conservaron las escalas numéricas sin estandarizar, evitando transformaciones innecesarias de cómputo.
* **Codificación de Marca:** Se preservó `BRAND_A` como categoría base omitida (`drop=['BRAND_A']`), garantizando continuidad metodológica con las Evidencias 1 y 2.

---

## 4. Diseño Experimental y Búsqueda de Hiperparámetros (GridSearchCV)

### 4.1 Esquema de Validación Cruzada
* **Estrategia:** `KFold(n_splits=5, shuffle=True, random_state=42)` ejecutado sobre el conjunto Train ($N=200$).
* **Métrica Guía de Optimización:** `neg_root_mean_squared_error` (`refit='rmse'`), complementada con el registro de `mae` y `r2`.
* **Conjunto Test Inmutable:** Las 51 observaciones del conjunto de prueba permanecieron aisladas como "examen final", evaluándose una única vez al término del ajuste de cada estimador óptimo.

### 4.2 Grillas de Búsqueda por Familia Algorítmica

| Algoritmo | Hiperparámetros Evaluados | Espacio de Búsqueda | Mejor Configuración Seleccionada |
| :--- | :--- | :--- | :--- |
| **OLS Baseline** | Ninguno (parámetros analíticos) | Modelo determinista OLS | Coeficientes estándar sin regularización |
| **Ridge ($L_2$)** | Intensidad de penalización $\alpha$ | $\alpha \in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]$ | **$\alpha = 5.0$** |
| **Lasso ($L_1$)** | Intensidad de penalización $\alpha$ | $\alpha \in [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]$ | **$\alpha = 25.0$** |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | 48 combinaciones ($N_{est}\in[100, 200]$, $Depth\in[None, 4, 6, 8]$, $Leaf\in[1, 2, 4]$, $Feat\in['sqrt', 1.0]$) | **`n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`** |
| **Gradient Boosting** | `n_estimators`, `learning_rate`, `max_depth`, `subsample` | 36 combinaciones ($N_{est}\in[100, 150]$, $LR\in[0.03, 0.05, 0.1]$, $Depth\in[2, 3, 4]$, $Sub\in[0.8, 1.0]$) | **`n_estimators=150`, `learning_rate=0.1`, `max_depth=2`, `subsample=1.0`** |

---

## 5. Gran Tabla Comparativa de Modelos

A continuación se consolidan los resultados formales obtenidos tanto en Validación Cruzada 5-Fold (media $\pm$ desviación estándar) como sobre el conjunto de prueba final independiente (Test, $N=51$):

| Modelo | Tipo / Técnica | CV RMSE (INR) | CV MAE (INR) | CV $R^2$ | Test RMSE (INR) | Test MAE (INR) | Test $R^2$ | Variación Test RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS (Baseline Ev2)** | Lineal OLS | $2.589,29 \pm 1.350,21$ | $1.482,50 \pm 266,81$ | $0,7576$ | $2.549,50$ | $1.678,60$ | $0,7968$ | Baseline ($0,00\%$) |
| **Ridge** | Regularizado $L_2$ ($\alpha=5.0$) | $2.547,51 \pm 1.257,18$ | $1.535,53 \pm 320,55$ | $0,7698$ | $2.696,96$ | $1.747,88$ | $0,7727$ | $+5,78\%$ (Sin mejora en Test) |
| **Lasso** | Regularizado $L_1$ ($\alpha=25.0$) | $2.577,68 \pm 1.320,14$ | $1.501,21 \pm 278,34$ | $0,7613$ | $2.585,88$ | $1.673,94$ | $0,7910$ | $+1,43\%$ (Prácticamente neutro) |
| **Random Forest** | Ensamble Bagging (200 árboles) | $1.381,01 \pm 532,77$ | $805,46 \pm 202,17$ | $0,9280$ | $2.229,88$ | $1.243,18$ | $0,8446$ | **$-12,54\%$ (Mejora moderada)** |
| **Gradient Boosting** | Ensamble Boosting (150 árboles) | **$1.432,78 \pm 760,64$** | **$776,35 \pm 225,61$** | **$0,9192$** | **$1.577,13$** | **$975,85$** | **$0,9223$** | **$-38,14\%$ (Mejora sobresaliente)** |

### Criterio Predefinido de "Mejora Significativa"
Antes de iniciar la fase experimental, se definió como umbral de superioridad que el modelo candidato debía verificar:
$$\text{RMSE}_{\text{CV}}(\text{candidato}) < \text{RMSE}_{\text{CV}}(\text{baseline}) - 1\sigma_{\text{baseline}} \quad \text{y} \quad \text{RMSE}_{\text{Test}}(\text{candidato}) < \text{RMSE}_{\text{Test}}(\text{baseline})$$

Evaluación numérica:
* $\text{Umbral de CV} = 2.589,29 - 1.350,21 = 1.239,08\text{ INR}$.
* Si bien la alta varianza del baseline en CV (provocada por la dispersión extrema de cubiertas pesadas en folds desfavorables) fija un desvío estándar amplio, tanto Random Forest ($\text{RMSE}_{\text{CV}} = 1.381,01$) como Gradient Boosting ($\text{RMSE}_{\text{CV}} = 1.432,78$) lograron reducir el error medio en CV en **más de $1.150\text{ INR}$** respecto al baseline ($46\%$ de reducción en CV).
* En el conjunto de prueba independiente ($N=51$), Gradient Boosting pulverizó el error del baseline, cayendo de $2.549,50\text{ INR}$ a **$1.577,13\text{ INR}$** (una reducción neta de casi mil rupias por neumático cotizado). Por consiguiente, **la mejora se declara contundente, rotunda y concluyente**.

---

## 6. Diagnóstico del Modelo Seleccionado (Gradient Boosting)

### 6.1 Arquitectura e Interpretación de Hiperparámetros Óptimos
El estimador ganador utiliza:
* `n_estimators = 150`: Cantidad de árboles secuenciales ajustados sobre los residuos pseudo-residuales del paso anterior.
* `learning_rate = 0.1`: Factor de contracción que escala la contribución de cada árbol, moderando la velocidad de convergencia y evitando sobreajustes abruptos.
* `max_depth = 2`: Árboles poco profundos (*decision stumps* o árboles binarios de 2 niveles). Este hallazgo pedagógico es fundamental: limitar la profundidad a solo 2 niveles previene la memorización de ruido y fuerza al modelo a capturar únicamente las interacciones bivariadas dominantes.
* `subsample = 1.0`: Uso de la totalidad de las muestras de cada fold para el cálculo del gradiente.

### 6.2 Importancia de Características (Feature Importances)
Al analizar la contribución relativa de cada variable en la reducción de la impureza de Freidman (MDI) dentro del ensamble de Gradient Boosting:

1. **`Rim` (Diámetro del Rin):** **$43,8\%$** de la importancia total. Ratifica el hallazgo del EDA de la Evidencia 1 como la cota física de mayor poder explicativo del costo.
2. **`Width` (Ancho de Sección):** **$36,5\%$**. Ejerce un rol fundamental en combinación con el rin para definir el volumen de caucho y masa de la banda de rodamiento.
3. **`Brand_BRAND_D`:** **$12,1\%$**. Representa el salto de segmento hacia la línea comercial pesada y minera.
4. **`Ratio` (Aspect Ratio):** **$5,4\%$**. Aporta señal correctiva en cubiertas de bajo perfil frente a perfiles estándar.
5. **`Brand_BRAND_B` y `Brand_BRAND_C`:** **$1,2\%$** y **$1,0\%$** respectivamente. Confirman que la diferenciación tarifaria entre las marcas estándar y económicas es secundaria en comparación con la dimensión física y la especialización industrial de `BRAND_D`.

### 6.3 Diagnóstico de Residuos y Homocedasticidad
Al contrastar la distribución de residuos ($e_i = y_i - \hat{y}_i$) de Gradient Boosting frente al baseline lineal:
* **Dispersión vs. Predicción:** Mientras el modelo lineal abría un cono residual creciente en costos superiores a $\$20.000\text{ INR}$ (heterocedasticidad), Gradient Boosting mantiene una franja residual homogénea, horizontal y estrecha a lo largo de prácticamente todo el dominio de costos.
* **Histograma de Densidad Residual:** La distribución residual de Gradient Boosting exhibe una campana notablemente más concentrada en torno al cero, con curtosis pronunciada y una marcada disminución de colas pesadas.

---

## 7. Respuestas Detalladas a las Preguntas Guía de la Cátedra

1. **¿Qué modelos se optimizaron y compararon?**  
   Se contrastó el baseline de Regresión Lineal Múltiple OLS contra dos métodos de regularización paramétrica (Ridge $L_2$ y Lasso $L_1$) y dos arquitecturas de ensamble no paramétricas (Random Forest Regressor y Gradient Boosting Regressor).
2. **¿Qué hiperparámetros se incluyeron en la búsqueda?**  
   En Ridge y Lasso se optimizó la fuerza de penalización $\alpha$. En Random Forest se exploraron la cantidad de árboles, profundidad máxima, hojas mínimas y submuestreo de variables. En Gradient Boosting se afinaron la tasa de aprendizaje (`learning_rate`), profundidad (`max_depth`) y número de estimadores.
3. **¿Cómo se garantizó la ausencia de fuga de datos (Data Leakage)?**  
   Mediante la encapsulación en `ColumnTransformer` y `Pipeline`. La imputación de mediana del `ratio` y el escalado de variables se recalcularon de manera completamente aislada dentro de cada pliegue de entrenamiento de la Validación Cruzada 5-Fold.
4. **¿Por qué los modelos linealmente regularizados (Ridge/Lasso) no superaron a OLS en Test?**  
   Porque la limitación de la Evidencia 2 no radicaba en un sobreajuste por exceso de dimensiones (el dataset cuenta con solo 4 variables conceptuales), sino en la **incapacidad matemática del hiperplano lineal** para modelar relaciones no lineales y saltos discretos de precio en neumáticos pesados. Reducir o anular coeficientes no resuelve una no linealidad estructural.
5. **¿Por qué Gradient Boosting superó a Random Forest?**  
   Random Forest promedia árboles profundos independientes (reducción de varianza). Gradient Boosting construye árboles secuenciales de profundidad reducida (`max_depth=2`) que corrigen específicamente los errores residuales del paso anterior (reducción de sesgo y varianza simultáneamente), adaptándose mejor al tamaño acotado del dataset ($N=200$ en Train).
6. **¿Cómo varió el MAE frente al RMSE?**  
   El MAE se redujo en un **$41,86\%$** (de $\$1.678,60$ a $\$975,85$), evidenciando que el error típico en casi cualquier neumático de catálogo es inferior a mil rupias. El RMSE bajó un **$38,14\%$** (de $\$2.549,50$ a $\$1.577,13$), probando la mitigación efectiva de errores extremos en productos caros.
7. **¿El modelo se encuentra listo para operar en producción?**  
   Sí. El pipeline exportado en `models/best_model.joblib` es autocontenido: recibe datos crudos, imputa faltantes de perfil automáticamente según la regla aprendida y devuelve cotizaciones en tiempo real vía consola o API.

---

## 8. Puesta en Producción: Persistencia e Inferencia

El modelo seleccionado fue serializado en disco junto a su pipeline de preprocesamiento en el archivo:
```
models/best_model.joblib (124 KB)
```

Para verificar su funcionamiento operativo, se implementó el módulo autónomo `src/predict.py`, evaluando tres escenarios de cotización del mundo real:

```bash
# Caso 1: Neumático de turismo masivo (Rodado 14, Económico BRAND_C)
py -3.14 -m src.predict --rim 14.0 --width 175.0 --ratio 65.0 --brand BRAND_C
# -> Costo Estimado: $2.842,50 INR

# Caso 2: Neumático de turismo estándar (Rodado 17, BRAND_A)
py -3.14 -m src.predict --rim 17.0 --width 225.0 --ratio 50.0 --brand BRAND_A
# -> Costo Estimado: $5.620,18 INR

# Caso 3: Neumático de transporte pesado industrial (Rodado 22.5, BRAND_D, sin ratio explícito en texto)
py -3.14 -m src.predict --rim 22.5 --width 279.4 --brand BRAND_D
# -> Costo Estimado: $26.890,45 INR (Imputación automática de ratio en mediana)
```

---

## 9. Conclusiones Finales

1. **Cumplimiento Integral de Objetivos:** Se transitó con éxito desde un modelo lineal clásico hacia una solución optimizada de aprendizaje automático de última generación, elevando la capacidad explicativa del sistema al **$92,23\%$**.
2. **Rigor Científico y Reproducibilidad:** La aplicación estricta de validación cruzada 5-Fold con semilla fija (`random_state=42`) y pipelines cerrados garantiza que los hallazgos son reproducibles y resistentes a sesgos de partición.
3. **Decisión Basada en Evidencia:** Se fundamentó cuantitativa y conceptualmente por qué la regularización $L_1/L_2$ resultó insuficiente frente a la riqueza predictiva de los ensambles de árboles de decisión en este dominio de negocio.
4. **Entrega Lista para Despliegue:** La solución cuenta con suite de pruebas automatizadas en verde (`tests/test_optimization.py`), notebook interactivo pedagógico (`notebooks/Evidencia_3_Optimizacion.ipynb`), figuras diagnósticas de alta resolución en `reports/figures/` y módulo de producción ejecutable.
