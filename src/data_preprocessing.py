"""Módulo de carga, depuración y extracción de especificaciones físicas de neumáticos.
Diseñado para la Evidencia de Aprendizaje N.º 2 - Desarrollo de IA (ISPC).
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_unit_cost(series: pd.Series) -> pd.Series:
    """Convierte cadenas monetarias con comas a float64 numérico limpio."""
    return (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('"', '', regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
    )


def parse_tire_specs(name: str) -> tuple:
    """Extrae las especificaciones físicas a partir del nombre comercial del neumático.
    
    Jerarquía de extracción:
    - Nivel 1: Ratio explícito en formato métrico (con o sin descriptor de velocidad Z).
    - Nivel 2: Cálculo geométrico exacto para neumáticos en formato de flotación.
    - Nivel 3: Nomenclaturas sin ratio determinable inequívocamente (asigna np.nan).
    
    Retorna: (width_mm, aspect_ratio_pct, rim_inches)
    """
    s = str(name).strip()

    # Nivel 2: Flotación en pulgadas (ej. '31 x 10.50 R 15 LT / 6PR')
    m_flot = re.search(r'^(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(?:Z\s*)?R\s*(\d+(?:\.\d+)?)', s, re.I)
    if m_flot:
        od_in = float(m_flot.group(1))
        w_in = float(m_flot.group(2))
        rim = float(m_flot.group(3))
        width_mm = w_in * 25.4
        ratio = ((od_in - rim) / (2.0 * w_in)) * 100.0
        return width_mm, ratio, rim

    # Nivel 1: Métrico con ratio explícito (ej. '145/70 R 12', '225/40 Z R 19')
    m_met = re.search(r'^(\d{3})/(\d{2})\s*(?:Z\s*)?R\s*(\d+(?:\.\d+)?)', s, re.I)
    if m_met:
        width_mm = float(m_met.group(1))
        ratio = float(m_met.group(2))
        rim = float(m_met.group(3))
        return width_mm, ratio, rim

    # Nivel 3A: Comercial ligero sin perfil explícito (ej. '155 R 12 C / 8PR')
    m_com = re.search(r'^(\d{3})\s*(?:Z\s*)?R\s*(\d+(?:\.\d+)?)', s, re.I)
    if m_com:
        width_mm = float(m_com.group(1))
        rim = float(m_com.group(2))
        return width_mm, np.nan, rim

    # Nivel 3B: Camión pesado en centésimas de pulgada (ej. '1000 R 20/18 PR')
    m_cam1 = re.search(r'^(\d{4})\s*(?:Z\s*)?R\s*(\d+(?:\.\d+)?)', s, re.I)
    if m_cam1:
        w_in = float(m_cam1.group(1)) / 100.0  # 1000 -> 10.00 pulgadas
        width_mm = w_in * 25.4
        rim = float(m_cam1.group(2))
        return width_mm, np.nan, rim

    # Nivel 3C: Camión pesado en pulgadas enteras (ej. '11 R 22.5/ 18 PR')
    m_cam2 = re.search(r'^(\d{1,2})\s*(?:Z\s*)?R\s*(\d+(?:\.\d+)?)', s, re.I)
    if m_cam2:
        w_in = float(m_cam2.group(1))
        width_mm = w_in * 25.4
        rim = float(m_cam2.group(2))
        return width_mm, np.nan, rim

    return np.nan, np.nan, np.nan


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Carga el dataset de stock, limpia el target y extrae las dimensiones físicas."""
    df = pd.read_csv(filepath)
    df['unit_cost_clean'] = clean_unit_cost(df['Unit Cost (INR)'])
    
    # Eliminación estricta de registros sin target supervisado
    df_clean = df.dropna(subset=['unit_cost_clean']).copy()
    
    specs = df_clean['Product Name'].apply(parse_tire_specs)
    df_clean['width'] = [s[0] for s in specs]
    df_clean['ratio'] = [s[1] for s in specs]
    df_clean['rim'] = [s[2] for s in specs]
    
    return df_clean


def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Divide los datos en Train y Test sobre las columnas de modelado."""
    feature_cols = ['rim', 'width', 'ratio', 'Brand']
    X = df[feature_cols].copy()
    y = df['unit_cost_clean'].copy()
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def impute_aspect_ratio(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Calcula la mediana de ratio exclusivamente sobre Train y la aplica a Train y Test."""
    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()
    
    train_median_ratio = float(X_train_imp['ratio'].dropna().median())
    
    X_train_imp['ratio'] = X_train_imp['ratio'].fillna(train_median_ratio)
    X_test_imp['ratio'] = X_test_imp['ratio'].fillna(train_median_ratio)
    
    return X_train_imp, X_test_imp, train_median_ratio


def encode_brands(X_train: pd.DataFrame, X_test: pd.DataFrame, reference_brand: str = 'BRAND_A'):
    """Aplica One-Hot Encoding fijando a reference_brand como categoría base omitida."""
    # Dummies completos
    X_train_enc = pd.get_dummies(X_train, columns=['Brand'], drop_first=False, dtype=float)
    X_test_enc = pd.get_dummies(X_test, columns=['Brand'], drop_first=False, dtype=float)
    
    # Alinear Test con las columnas de Train
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0.0)
    
    # Descartar la categoría de referencia de forma controlada
    ref_col = f"Brand_{reference_brand}"
    if ref_col in X_train_enc.columns:
        X_train_enc = X_train_enc.drop(columns=[ref_col])
        X_test_enc = X_test_enc.drop(columns=[ref_col])
        
    return X_train_enc, X_test_enc
