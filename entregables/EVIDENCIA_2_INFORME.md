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

En el marco de la **Evidencia de Aprendizaje N.º 2**, se implementó un sistema de aprendizaje supervisado basado en el paradigma de **Regresión Lineal**, orientado a predecir el costo unitario de catálogo (`Unit Cost (INR)`) de neumáticos a partir de sus especificaciones físicas y su fabricante.

El desarrollo da continuidad directa al Análisis Exploratorio de Datos (EDA) de la **Evidencia N.º 1**, donde se demostró la existencia de correlaciones lineales significativas entre las dimensiones físicas del producto y su costo. Se diseñó un flujo de ingeniería libre de fuga de información (*data leakage*), dividiendo el catálogo en un 80% de entrenamiento (200 observaciones) y un 20% de prueba (51 observaciones) con semilla reproducible (`random_state=42`).

Se contrastó un modelo base de **Regresión Lineal Simple** ($\text{Rim} \rightarrow \text{Cost}$) frente a un modelo integrador de **Regresión Lineal Múltiple** ($\text{Rim} + \text{Width} + \text{Ratio} + \text{Brand} \rightarrow \text{Cost}$). Los resultados confirman que la incorporación de la geometría completa y la marca reduce el error absoluto medio de $2.795,00$ INR a $1.678,60$ INR y eleva la varianza explicada ($R^2$) del $54,21\%$ a casi el $80\%$ ($79,68\%$), demostrando una sólida capacidad de generalización ante datos no vistos.

---

## 2. Antecedentes y Enlace con la Evidencia N.º 1 (EDA)

En la etapa previa, el equipo analizó las tablas `Current Stock` (253 productos), `Sales` (16.999 transacciones) y `Supplier` (4 proveedores). Los hallazgos cuantitativos clave determinaron:
1. **Especificaciones como variables predictoras:** El campo de texto `Product Name` codifica el ancho de sección, el perfil y el diámetro del rin de cada neumático.
2. **Jerarquía de correlaciones lineales:** El diámetro del rin (`Rim`) evidenció la correlación individual más alta con el costo ($r = 0,76$), seguido por el ancho de sección (`Width`, $r = 0,44$). La relación de aspecto (`Aspect Ratio`) exhibió una relación moderada-baja ($r = 0,36$), mientras que el stock actual no aportó señal física ($r = 0,08$).
3. **Efecto de marca:** Se constató una marcada dispersión tarifaria entre fabricantes, donde `BRAND_D` se posiciona en el segmento industrial/pesado de alto costo, mientras `BRAND_C` concentra la oferta económica masiva.
4. **Viabilidad técnica:** El EDA concluyó textualmente que esta estructura dimensional conformaba una *"buena señal para regresión"*.

---

## 3. Formulación del Problema y Justificación del Dataset

* **Variable Objetivo (Target):** `Unit Cost (INR)`, continua en escala de razón, correspondiente al costo unitario del producto del catálogo.
* **Dataset Seleccionado:** `DS_Data_Set_Current_Stock.csv`.
* **Criterio de Diseño del Grano de Datos:** Al modelar el costo intrínseco del neumático como bien manufacturado, la unidad de análisis debe ser el producto individual de catálogo (SKU). Utilizar la tabla transaccional `Sales` distorsionaría el modelo, ya que ponderaría los costos por la frecuencia de compra de los clientes e introduciría variaciones comerciales ajenas a la fabricación, como niveles de descuento por volumen (`Price Level`) y devoluciones.

---

## 4. Metodología de Preprocesamiento e Ingeniería de Features

### 4.1 Limpieza del Target
Se removieron máscaras textuales de formato monetario. De los 253 registros iniciales, se eliminaron estrictamente las **2 observaciones con costo nulo** (índices 251 y 252, productos `P0166` y `P0119`). En aprendizaje supervisado no se imputa la variable objetivo; entrenar o evaluar sobre valores sintéticos degrada la validez empírica. La población supervisada final quedó fijada en **251 observaciones**.

### 4.2 Parser Taxonómico en Tres Niveles
A partir de `Product Name`, se estructuró un extractor de expresiones regulares que cubre el 100% del inventario:
* **Nivel 1 (Ratio Explícito - 218 casos):** En neumáticos métricos estándar (ej. `145/70 R 12`, `225/40 Z R 19`), se extrae directamente el ancho en mm, el diámetro del rin en pulgadas y el ratio numérico tras la barra `/`.
* **Nivel 2 (Flotación en Pulgadas - 11 casos, ej. `31 x 10.50 R 15`):** Convierte el ancho a milímetros ($10.50 \times 25.4 = 266.7\text{ mm}$) y deduce geométricamente el perfil real a partir del diámetro exterior ($OD$) y el rin:
  $$H = \frac{OD - Rim}{2} = \frac{31 - 15}{2} = 8.0\text{ in} \implies \text{Aspect Ratio} = \left(\frac{8.0}{10.50}\right) \times 100 \approx 76,19\%$$
* **Nivel 3 (Nomenclaturas sin ratio determinable - 24 casos):** Comerciales ligeros tipo C (`155 R 12 C`) y transporte pesado (`1000 R 20`, `11 R 22.5`). El parser extrae `Width` (reconociendo que `1000` representa $10.00$ pulgadas y no 1000 mm) y `Rim`, asignando estrictamente `NaN` en `Aspect Ratio` para evitar supuestos fijos arbitrarios.

### 4.3 Imputación sin Fuga de Datos (Data Leakage)
> **Principio de Diseño:** Cualquier estadístico que resuma la distribución de los datos (media, mediana, varianza) constituye un parámetro del modelo. Calcularlo sobre el dataset completo antes de particionar filtra información de prueba hacia el entrenamiento (*data leakage*), produciendo métricas artificialmente optimistas.

Para garantizar rigor metodológico:
1. Se ejecutó primero la división Train/Test.
2. Se calculó la mediana de `Aspect Ratio` **exclusivamente sobre el conjunto de entrenamiento** ($\text{mediana}_{\text{Train}} = 65,00\%$).
3. Dicho valor se utilizó para imputar los faltantes de Train y se aplicó posteriormente sobre los faltantes de Test, simulando exactamente cómo operaría el modelo ante datos nuevos en producción.

### 4.4 Codificación de Marca (One-Hot Encoding con Base de Referencia)
> **Principio de Diseño:** En un modelo de regresión lineal múltiple con término independiente ($\beta_0$), la inclusión de todas las categorías binarias genera multicolinealidad perfecta (la suma de las dummies es idéntica al vector de unos del intercepto, provocando que la matriz de diseño sea no invertible).

Por esta razón:
1. Se fijó explícitamente a **`BRAND_A`** como la categoría de referencia omitida (`drop_first`).
2. El intercepto $\beta_0$ absorbe el costo base de un neumático `BRAND_A`.
3. Los coeficientes de `BRAND_B`, `BRAND_C` y `BRAND_D` se interpretan como diferenciales directos de costo respecto a `BRAND_A`.
4. El conjunto Test se alineó con las columnas exactas de Train mediante `reindex(fill_value=0.0)`.

---

## 5. Partición Train / Test y Decisión sobre Escalado

* **Configuración:** `train_test_split(test_size=0.20, random_state=42)`.
* **Distribución Resultante:**
  * **Entrenamiento (Train):** **200 observaciones** ($79,68\%$).
  * **Prueba (Test):** **51 observaciones** ($20,32\%$).
* **Justificación de la No Estandarización (Invarianza OLS):** A diferencia de algoritmos basados en distancias (KNN, K-Means) o regularización penalizada (Ridge, Lasso), la solución analítica de Mínimos Cuadrados Ordinarios (OLS) es invariante ante transformaciones afines de escala: no altera el hiperplano óptimo, las predicciones ni el $R^2$. Mantener las unidades originales (pulgadas, milímetros, porcentajes) permite interpretar cada coeficiente directamente en unidades monetarias (INR), preservando la transparencia pedagógica del Colab de la cátedra.

---

## 6. Modelado y Coeficientes Obtenidos

### 6.1 Modelo 1 — Regresión Lineal Simple (Baseline)
Utiliza únicamente el predictor individual con mayor correlación:
$$\widehat{\text{Cost}} = \beta_0 + \beta_1 \cdot \text{Rim}$$

* **Parámetros Ajustados:**
  * Intercepto ($\beta_0$): **$-22.783,89$ INR**
  * Coeficiente $\text{Rim}$ ($\beta_1$): **$+1.957,17$ INR/pulgada**
* **Lectura Técnica:** Por cada pulgada adicional en el diámetro del rin, el costo unitario esperado se incrementa en promedio \$1.957,17 INR.

### 6.2 Modelo 2 — Regresión Lineal Múltiple (Integrador)
Incorpora las tres cotas geométricas y el efecto de fabricante (Base: `BRAND_A`):
$$\widehat{\text{Cost}} = \beta_0 + \beta_1 \text{Rim} + \beta_2 \text{Width} + \beta_3 \text{Ratio} + \beta_4 B_B + \beta_5 B_C + \beta_6 B_D$$

* **Parámetros Ajustados:**
  * Intercepto ($\beta_0$): **$-34.926,44$ INR**
  * $\text{Rim}$ ($\beta_1$): **$+1.498,42$ INR/pulgada**
  * $\text{Width}$ ($\beta_2$): **$+24,09$ INR/mm**
  * $\text{Ratio}$ ($\beta_3$): **$+223,70$ INR/%**
  * $\text{Brand\_BRAND\_B}$ ($\beta_4$): **$+468,73$ INR**
  * $\text{Brand\_BRAND\_C}$ ($\beta_5$): **$-1.070,51$ INR**
  * $\text{Brand\_BRAND\_D}$ ($\beta_6$): **$+4.821,93$ INR**
* **Lectura Técnica de Coeficientes:**
  * Manteniendo constantes las demás variables, cada pulgada extra de rin añade \$1.498 INR y cada milímetro de ancho añade \$24 INR al costo esperado.
  * El coeficiente negativo de `BRAND_C` ($-1.070,51$ INR) confirma cuantitativamente que es una marca económica que cuesta en promedio \$1.070 menos que `BRAND_A` a igualdad de medidas.
  * El coeficiente marcadamente positivo de `BRAND_D` ($+4.821,93$ INR) refleja la prima de mercado impuesta por su especialización en cubiertas pesadas e industriales.

---

## 7. Evaluación de Desempeño sobre Datos de Prueba (Test, N=51)

Las métricas se evaluaron sobre las 51 observaciones de prueba independientes que el modelo nunca observó durante el ajuste:

| Métrica Formal | Modelo Simple (`Rim`) | Modelo Múltiple (`Specs + Brand`) | Mejora Relativa |
| :--- | :---: | :---: | :---: |
| **MAE (Error Absoluto Medio)** | $2.795,00$ INR | **$1.678,60$ INR** | **$-39,94\%$ (Mayor precisión general)** |
| **MSE (Error Cuadrático Medio)** | $14.651.537,71$ | **$6.499.975,49$** | **$-55,64\%$ (Menor varianza residual)** |
| **RMSE (Raíz Error Cuadrático)** | $3.827,73$ INR | **$2.549,50$ INR** | **$-33,39\%$ (Menor impacto de atípicos)** |
| **$R^2$ (Coef. de Determinación)** | $0,5421$ ($54,21\%$) | **$0,7968$ ($79,68\%$)** | **$+46,98\%$ en varianza explicada** |

> **Interpretación Pedagógica de MAE frente a RMSE:**  
> MAE mide la magnitud media del error sin ponderación: en promedio, el modelo múltiple se equivoca por \$1.678,60 INR. RMSE, en cambio, eleva los residuos al cuadrado antes de promediar, penalizando de forma cuadrática las desviaciones severas. La brecha entre RMSE (\$2.549,50) y MAE (\$1.678,60) indica que, aunque la mayor parte del catálogo se predice con exactitud, subsiste un conjunto reducido de observaciones atípicas con errores mayores.

---

## 8. Diagnóstico de Generalización y Análisis de Residuos

### 8.1 Desempeño Train vs. Test
Para evaluar la capacidad de generalización:
* **Train:** $R^2 = 0,8966$ | $\text{RMSE} = 1.832,23$ INR
* **Test:** $R^2 = 0,7968$ | $\text{RMSE} = 2.549,50$ INR

*Diagnóstico:* La proximidad de métricas entre entrenamiento y prueba descarta indicios de sobreajuste severo (*overfitting*) y valida que el modelo generaliza consistentemente ante datos nuevos.

### 8.2 Inspección de Gráficos Diagnósticos
1. **Recta Simple:** Visualiza la tendencia creciente del costo respecto al rin, evidenciando limitaciones para capturar la variabilidad de cubiertas con diámetros comunes (15 a 17 pulgadas).
2. **Valores Reales vs. Predichos:** En el modelo múltiple, las muestras se agrupan estrechamente en torno a la línea de identidad $y = x$, demostrando un salto cualitativo respecto al baseline.
3. **Gráfico de Residuos:** Los residuos ($e_i = y_i - \hat{y}_i$) se dispersan de forma homocedástica y aleatoria en torno a cero para el 90% de los productos (costos inferiores a 15.000 INR). La dispersión se incrementa únicamente en productos premium de costo muy elevado (más de 25.000 INR), confirmando la presencia de heterocedasticidad en la cola superior.

---

## 9. Respuestas a las Preguntas Guía de la Cátedra

1. **¿Qué variable predecimos?** El costo unitario (`Unit Cost (INR)`) del producto de catálogo.
2. **¿Qué variables ayudan a explicarla?** El diámetro del rin (`Rim`), el ancho (`Width`), la relación de aspecto (`Aspect Ratio`) y el fabricante (`Brand`).
3. **¿Qué tan bien funciona el baseline?** Alcanza un $R^2 = 54,21\%$, probando que el rin es la dimensión individual más influyente, aunque deja un error promedio elevado (\$2.795 INR).
4. **¿Qué ocurre al incorporar Width, Ratio y Brand?** El $R^2$ asciende al $79,68\%$ y el error MAE se reduce en un $40\%$.
5. **¿Cómo cambia el MAE?** Cae de $2.795,00$ INR a $1.678,60$ INR.
6. **¿Cómo cambia el RMSE?** Desciende de $3.827,73$ INR a $2.549,50$ INR, reduciendo la severidad de errores extremos.
7. **¿Cómo cambia el $R^2$?** Se incrementa en 25 puntos porcentuales netos (de $0,5421$ a $0,7968$).
8. **¿Existen indicios de errores sistemáticos?** No en la masa general del catálogo; solo se aprecia mayor dispersión residual en productos premium de cola derecha.
9. **¿El modelo generaliza razonablemente a Test?** Sí, el rendimiento en prueba replica coherentemente el nivel de acierto de entrenamiento.
10. **¿Qué limitaciones quedan para la siguiente entrega?** La linealidad estricta no modela interacciones no lineales ni la asimetría de precios extremos.

---

## 10. Roadmap hacia la Evidencia N.º 3 (Trabajo Futuro)

En conformidad con las pautas de la cátedra, las técnicas de optimización avanzada quedan reservadas para la Tercera Evidencia:
* **Transformación Logarítmica:** Evaluar $\ln(\text{Unit Cost})$ para estabilizar la varianza residual en la cola superior de precios.
* **Regularización Lineal:** Implementar modelos Ridge ($L_2$) y Lasso ($L_1$) para mitigar la correlación geométrica entre ancho y rin.
* **Modelos de Ensamble:** Explorar Random Forest Regressor y Gradient Boosting para capturar relaciones no lineales complejas.
* **Ajuste Fino de Hiperparámetros:** Búsqueda sistemática mediante validación cruzada (*GridSearchCV*).
