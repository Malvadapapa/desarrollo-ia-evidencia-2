# INFORME TÉCNICO: EVIDENCIA DE APRENDIZAJE N.º 2
## Espacio Curricular: Desarrollo de Inteligencia Artificial
### Tecnicatura Superior en Desarrollo de Software — ISPC

**Docente:** Cra. Sol Figueroa  
**Equipo de Desarrollo:**
* Vargas, Cristian
* Quinteros, Karina
* Zárate, Laura

---

## 1. Resumen Ejecutivo

En el marco de la **Evidencia de Aprendizaje N.º 2**, se implementó un sistema predictivo de aprendizaje automático supervisado bajo el paradigma de **Regresión Lineal**, orientado a estimar el costo unitario de catálogo (`Unit Cost (INR)`) de neumáticos a partir de sus especificaciones físicas y su fabricante.

El desarrollo da continuidad directa al Análisis Exploratorio de Datos (EDA) entregado en la **Evidencia N.º 1**, donde se demostró la existencia de correlaciones lineales significativas entre las dimensiones del producto y su costo. Se diseñó un flujo de preparación riguroso y libre de fuga de información (*data leakage*), dividiendo el catálogo en un 80% de entrenamiento (200 observaciones) y un 20% de prueba (51 observaciones) con semilla reproducible (`random_state=42`).

Se contrastó un modelo base de **Regresión Lineal Simple** ($\text{Rim} \rightarrow \text{Cost}$) frente a un modelo integrador de **Regresión Lineal Múltiple** ($\text{Rim} + \text{Width} + \text{Ratio} + \text{Brand} \rightarrow \text{Cost}$). Los resultados confirman que la incorporación de la geometría completa y la marca reduce el error absoluto medio de $2.795,00$ INR a $1.678,60$ INR y eleva la varianza explicada ($R^2$) del $54,21\%$ a casi el $80\%$ ($79,68\%$), demostrando una sólida capacidad de generalización ante datos no vistos.

---

## 2. Antecedentes y Enlace con la Evidencia N.º 1 (EDA)

En la etapa previa, el equipo analizó las tablas `Current Stock` (253 productos), `Sales` (16.999 transacciones) y `Supplier` (4 proveedores). Los hallazgos cuantitativos clave asentados en el informe de Evidencia 1 determinaron:
1. **Especificaciones como variables predictoras:** El campo de texto `Product Name` codifica el ancho de sección, el perfil y el diámetro del rin de cada neumático.
2. **Jerarquía de correlaciones lineales:** El diámetro del rin (`Rim`) evidenció la correlación individual más alta con el costo ($r = 0,76$), seguido por el ancho de sección (`Width`, $r = 0,44$). La relación de aspecto (`Aspect Ratio`) exhibió una relación moderada-baja ($r = 0,36$), mientras que el stock actual (`Current Stock Qty`) no aportó señal física ($r = 0,08$).
3. **Efecto de marca:** Se constató una marcada dispersión tarifaria entre fabricantes, donde `BRAND_D` se posiciona en el segmento industrial/pesado de alto costo, mientras `BRAND_C` concentra la oferta económica masiva.
4. **Viabilidad técnica:** El EDA concluyó textualmente que esta estructura dimensional conformaba una *"buena señal para regresión"*.

---

## 3. Formulación del Problema y Justificación del Dataset

* **Variable Objetivo (Target):** `Unit Cost (INR)`, continua en escala de razón, correspondiente al costo unitario del producto del catálogo.
* **Dataset Seleccionado:** `DS_Data_Set_Current_Stock.csv`.
* **Justificación Metodológica:** Al modelar el costo intrínseco del producto, la unidad de análisis debe ser el producto individual (SKU). Utilizar la tabla transaccional `Sales` distorsionaría el modelo, ya que ponderaría los costos por la frecuencia de compra de los clientes e introduciría variaciones comerciales derivadas de los canales de descuento (`Price Level`) y devoluciones.

---

## 4. Ingeniería de Características y Tratamiento de Faltantes

### 4.1 Limpieza del Target
Se removieron máscaras textuales de formato monetario. De los 253 registros iniciales, se eliminaron estrictamente las **2 observaciones con costo nulo** (índices 251 y 252, productos `P0166` y `P0119`). No se imputó la variable dependiente para no inducir sesgos sintéticos. La población supervisada final quedó en **251 observaciones**.

### 4.2 Parser de Especificaciones Físicas
Se implementó un parser de expresiones regulares estructurado en tres niveles para procesar el 100% del inventario:
* **Nivel 1 (Ratio Explícito - 206 casos métricos estándar + 12 casos con código Z):** Extrae directamente `Width` en mm, `Aspect Ratio` en % tras la barra `/`, y `Rim` en pulgadas.
* **Nivel 2 (Flotación en Pulgadas - 11 casos, ej. `31 x 10.50 R 15`):** Convierte el ancho a milímetros ($10.50 \times 25.4 = 266.7\text{ mm}$) y deduce geométricamente el ratio real a partir del diámetro exterior ($OD$) y el rin:
  $$H = \frac{OD - Rim}{2} = \frac{31 - 15}{2} = 8.0\text{ in} \implies \text{Aspect Ratio} = \left(\frac{8.0}{10.50}\right) \times 100 \approx 76,19\%$$
* **Nivel 3 (Nomenclaturas sin ratio deducible - 24 casos):** Comerciales ligeros tipo C (`155 R 12 C`) y transporte pesado (`1000 R 20`, `11 R 22.5`). El parser extrae correctamente `Width` (contemplando que `1000` denota $10.00$ pulgadas y no 1000 mm) y `Rim`, asignando estrictamente `NaN` en `Aspect Ratio` para evitar supuestos arbitrarios.

### 4.3 Imputación sin Fuga de Datos (Data Leakage)
Siguiendo las mejores prácticas de la cátedra:
1. Se ejecutó primero la división Train/Test.
2. Se calculó la mediana de `Aspect Ratio` **exclusivamente sobre el conjunto de entrenamiento** ($\text{mediana}_{\text{Train}} = 65,00\%$).
3. Dicho valor se utilizó para imputar los faltantes de Train y se aplicó posteriormente a los faltantes de Test, garantizando total independencia muestral.

### 4.4 Codificación de Marca (One-Hot Encoding)
Se aplicó `pd.get_dummies` fijando explícitamente a **`BRAND_A`** como categoría de referencia (omitida). El conjunto Test se alineó con las columnas exactas de Train mediante `reindex(fill_value=0.0)`, produciendo las variables binarias `Brand_BRAND_B`, `Brand_BRAND_C` y `Brand_BRAND_D`.

---

## 5. Partición Train / Test y Configuración Experimental

* **Configuración:** `train_test_split(test_size=0.20, random_state=42)`.
* **Distribución Resultante:**
  * **Entrenamiento (Train):** **200 observaciones** ($79,68\%$).
  * **Prueba (Test):** **51 observaciones** ($20,32\%$).
* **Rigor Estadístico:** Debido a la indivisibilidad de 251 observaciones discretas ($251 \times 0.20 = 50.2$), Scikit-Learn asigna 51 registros a test. La semilla `random_state=42` garantiza la exacta reproducibilidad de todas las operaciones.
* **Escalado:** Se prescindió intencionalmente de `StandardScaler` en la entrega para preservar la interpretabilidad directa de los parámetros en unidades monetarias (INR), manteniendo el estilo procedimental del Colab de la docente.

---

## 6. Modelado y Ecuaciones Deducidas

### 6.1 Modelo 1 — Regresión Lineal Simple (Baseline)
Utiliza únicamente el predictor de mayor peso individual (`Rim`):
$$\widehat{\text{Cost}} = \beta_0 + \beta_1 \cdot \text{Rim}$$

* **Parámetros Ajustados (sobre Train):**
  * Intercepto ($\beta_0$): **$-22.783,89$ INR**
  * Coeficiente $\text{Rim}$ ($\beta_1$): **$+1.957,17$ INR/pulgada**
* **Interpretación:** Por cada pulgada de incremento en el diámetro de la llanta, el costo del neumático aumenta en promedio casi 2.000 INR.

### 6.2 Modelo 2 — Regresión Lineal Múltiple (Integrador)
Incorpora las tres dimensiones físicas y la marca del neumático:
$$\widehat{\text{Cost}} = \beta_0 + \beta_1 \text{Rim} + \beta_2 \text{Width} + \beta_3 \text{Ratio} + \beta_4 B_B + \beta_5 B_C + \beta_6 B_D$$

* **Parámetros Ajustados (sobre Train):**
  * Intercepto ($\beta_0$): **$-34.926,44$ INR**
  * $\text{Rim}$ ($\beta_1$): **$+1.498,42$ INR/pulgada**
  * $\text{Width}$ ($\beta_2$): **$+24,09$ INR/mm**
  * $\text{Ratio}$ ($\beta_3$): **$+223,70$ INR/%**
  * $\text{Brand\_BRAND\_B}$ ($\beta_4$): **$+468,73$ INR**
  * $\text{Brand\_BRAND\_C}$ ($\beta_5$): **$-1.070,51$ INR** (más económica que `BRAND_A`)
  * $\text{Brand\_BRAND\_D}$ ($\beta_6$): **$+4.821,93$ INR** (fuerte prima por segmento industrial)

---

## 7. Evaluación de Desempeño sobre Datos de Prueba (Test)

Las métricas se evaluaron sobre las 51 observaciones de prueba que nunca fueron vistas por los algoritmos durante el ajuste:

| Métrica Formal | Modelo Simple (`Rim`) | Modelo Múltiple (`Specs + Brand`) | Variación Relativa |
| :--- | :---: | :---: | :---: |
| **MAE (Error Absoluto Medio)** | $2.795,00$ INR | **$1.678,60$ INR** | **$-39,94\%$ (Mejora drástica)** |
| **MSE (Error Cuadrático Medio)** | $14.651.537,71$ | **$6.499.975,49$** | **$-55,64\%$ (Menos dispersión)** |
| **RMSE (Raíz del Error Cuadrático)** | $3.827,73$ INR | **$2.549,50$ INR** | **$-33,39\%$ (Menos fallos extremos)** |
| **$R^2$ (Coef. de Determinación)** | $0,5421$ ($54,21\%$) | **$0,7968$ ($79,68\%$)** | **$+46,98\%$ en varianza explicada** |

---

## 8. Diagnóstico de Generalización y Análisis de Residuos

### 8.1 Desempeño Train vs. Test
Para evaluar la capacidad de generalización sin emitir conclusiones axiomáticas prematuras, se contrastó el rendimiento del Modelo Múltiple en ambos conjuntos:
* **Train:** $R^2 = 0,8966$ | $\text{RMSE} = 1.832,23$ INR
* **Test:** $R^2 = 0,7968$ | $\text{RMSE} = 2.549,50$ INR

*Diagnóstico:* La proximidad de las métricas entre ambos subconjuntos evidencia un modelo bien balanceado. No se observan indicios de sobreajuste severo (*overfitting*); el modelo generaliza consistentemente sobre muestras independientes de catálogo.

### 8.2 Inspección de Gráficos Diagnósticos
1. **Recta Simple:** La recta estimada confirma visualmente la tendencia ascendente observada en el EDA, aunque muestra dispersión en rines de 15 a 17 pulgadas.
2. **Valores Reales vs. Predichos:** Las observaciones del modelo múltiple se concentran con notable cercanía alrededor de la línea de identidad ($y = x$), superando con claridad al modelo simple.
3. **Gráfico de Residuos:** La dispersión de $e_i = y_i - \hat{y}_i$ frente a los valores predichos $\hat{y}_i$ se distribuye homogéneamente alrededor de cero para la gran masa de productos de gama media (menos de 15.000 INR). Se advierte una mayor amplitud residual exclusivamente en productos premium de costo muy elevado (más de 25.000 INR), fenómeno típico de variables de precio asimétricas.

---

## 9. Respuestas a las Preguntas Guía de la Cátedra

1. **¿Qué variable estamos intentando predecir?**  
   El costo unitario (`Unit Cost (INR)`) de los neumáticos en el inventario.
2. **¿Qué variables ayudan a explicarla?**  
   El diámetro del rin (`Rim`), el ancho (`Width`), la relación de aspecto (`Aspect Ratio`) y el fabricante (`Brand`).
3. **¿Qué tan bien funciona el baseline?**  
   Alcanza un $R^2$ de $54,21\%$, probando que el rin es la dimensión individual más influyente, aunque genera un error promedio alto ($2.795$ INR).
4. **¿Qué ocurre al incorporar Width, Ratio y Brand?**  
   El poder explicativo asciende a casi el $80\%$ ($R^2 = 0,7968$) y el error medio desciende en un $40\%$.
5. **¿Cómo cambia el MAE?**  
   Se reduce de $2.795,00$ INR a $1.678,60$ INR.
6. **¿Cómo cambia el RMSE?**  
   Disminuye de $3.827,73$ INR a $2.549,50$ INR, reduciendo la severidad de los errores atípicos.
7. **¿Cómo cambia el $R^2$?**  
   Pasa de $0,5421$ a $0,7968$.
8. **¿Existen indicios de errores sistemáticos?**  
   No en el segmento general. Sin embargo, en el extremo superior de precios se evidencia heterocedasticidad marginal por el sesgo natural de la variable monetaria.
9. **¿El modelo generaliza razonablemente al conjunto de test?**  
   Sí. El error se mantiene acotado y coherente con el de entrenamiento.
10. **¿Qué limitaciones quedan para la siguiente evidencia?**  
    El supuesto de linealidad rígida no captura interacciones complejas ni la asimetría de los costos premium.

---

## 10. Roadmap hacia la Evidencia N.º 3 (Trabajo Futuro)

En conformidad con las pautas de la cátedra, las etapas de optimización y ajuste fino quedan formalmente planteadas para la Tercera Evidencia:
* **Transformación Logarítmica:** Evaluar $\ln(\text{Unit Cost})$ para normalizar la varianza residual en modelos premium.
* **Regularización Lineal:** Implementar modelos Ridge ($L_2$) y Lasso ($L_1$) para penalizar coeficientes ante posible colinealidad entre ancho y rin.
* **Modelos de Ensamble:** Explorar algoritmos de ensamble basados en árboles (*Random Forest Regressor* y *Gradient Boosting*) para modelar interacciones no lineales.
* **Búsqueda de Hiperparámetros:** Optimización sistemática mediante validación cruzada (*GridSearchCV*).
