"""Módulo de ingeniería de pipelines, validación cruzada y optimización de hiperparámetros.
Diseñado para la Evidencia de Aprendizaje N.º 3 - Desarrollo de IA (ISPC).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_NUM_COLS: List[str] = ['rim', 'width', 'ratio']
FEATURE_CAT_COLS: List[str] = ['Brand']
BRAND_CATEGORIES: List[str] = ['BRAND_A', 'BRAND_B', 'BRAND_C', 'BRAND_D']


def build_preprocessor(with_scaler: bool = False, reference_brand: str = 'BRAND_A') -> ColumnTransformer:
    """Construye un ColumnTransformer para imputación y codificación libre de data leakage.
    
    Parámetros:
    -----------
    with_scaler : bool
        Si es True, aplica StandardScaler a las variables numéricas (indispensable para
        modelos lineales regularizados como Ridge y Lasso). Si es False, mantiene la escala
        original (para modelos basados en árboles y OLS).
    reference_brand : str
        Categoría omitida para One-Hot Encoding para evitar multicolinealidad perfecta
        (por defecto 'BRAND_A', manteniendo consistencia con Evidencia 2).
    
    Retorna:
    --------
    ColumnTransformer configurado.
    """
    if with_scaler:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(
            categories=[BRAND_CATEGORIES],
            drop=[reference_brand],
            sparse_output=False,
            handle_unknown='ignore'
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, FEATURE_NUM_COLS),
            ('cat', cat_pipeline, FEATURE_CAT_COLS)
        ],
        remainder='drop'
    )
    return preprocessor


def build_pipeline(
    regressor: BaseEstimator,
    with_scaler: bool = False,
    reference_brand: str = 'BRAND_A',
    log_target: bool = False
) -> BaseEstimator:
    """Construye un Pipeline integrado con preprocesamiento y modelo estimador.
    
    Si log_target es True, envuelve el pipeline en un TransformedTargetRegressor con log1p y expm1
    para modelar la asimetría y heterocedasticidad en la variable objetivo.
    """
    preprocessor = build_preprocessor(with_scaler=with_scaler, reference_brand=reference_brand)
    
    base_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    if log_target:
        return TransformedTargetRegressor(
            regressor=base_pipe,
            func=np.log1p,
            inverse_func=np.expm1
        )
    return base_pipe


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Obtiene los nombres de las columnas transformadas luego del ajuste."""
    num_names = list(FEATURE_NUM_COLS)
    cat_step = preprocessor.named_transformers_['cat']
    ohe = cat_step.named_steps['onehot']
    cat_names = list(ohe.get_feature_names_out(FEATURE_CAT_COLS))
    return num_names + cat_names


def cross_validate_pipeline(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Ejecuta validación cruzada K-Fold evaluando RMSE, MAE y R²."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=kf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    cv_rmse_folds = -scores['test_rmse']
    cv_mae_folds = -scores['test_mae']
    cv_r2_folds = scores['test_r2']
    
    train_rmse_folds = -scores['train_rmse']
    train_mae_folds = -scores['train_mae']
    train_r2_folds = scores['train_r2']
    
    return {
        'cv_rmse_mean': float(cv_rmse_folds.mean()),
        'cv_rmse_std': float(cv_rmse_folds.std()),
        'cv_mae_mean': float(cv_mae_folds.mean()),
        'cv_mae_std': float(cv_mae_folds.std()),
        'cv_r2_mean': float(cv_r2_folds.mean()),
        'cv_r2_std': float(cv_r2_folds.std()),
        'train_rmse_mean': float(train_rmse_folds.mean()),
        'train_r2_mean': float(train_r2_folds.mean()),
        'cv_rmse_folds': cv_rmse_folds,
        'cv_r2_folds': cv_r2_folds,
        'cv_mae_folds': cv_mae_folds
    }


def get_default_search_spaces() -> Dict[str, Dict[str, Any]]:
    """Define los espacios de búsqueda sistemática para cada familia algorítmica."""
    return {
        'Ridge': {
            'estimator_factory': lambda: build_pipeline(Ridge(), with_scaler=True),
            'param_grid': {
                'regressor__alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
            }
        },
        'Random Forest': {
            'estimator_factory': lambda: build_pipeline(RandomForestRegressor(random_state=42), with_scaler=False),
            'param_grid': {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [None, 4, 6, 8],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__max_features': ['sqrt', 1.0]
            }
        }
    }


def paired_fold_comparison(baseline_folds: np.ndarray, candidate_folds: np.ndarray) -> Dict[str, Any]:
    """Compara dos modelos fold a fold (mismos splits de KFold).

    Devuelve la diferencia media de RMSE (baseline - candidato), su desvío
    y en cuántos folds el candidato tuvo menor error.
    """
    base = np.asarray(baseline_folds, dtype=float)
    cand = np.asarray(candidate_folds, dtype=float)
    if base.shape != cand.shape:
        raise ValueError("Ambos modelos deben evaluarse sobre los mismos folds")
    diff = base - cand
    return {
        'diff_per_fold': diff,
        'mean_diff': float(diff.mean()),
        'std_diff': float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        'wins': int((diff > 0).sum()),
        'n_folds': int(len(diff))
    }


def run_grid_search(
    pipeline: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> GridSearchCV:
    """Ejecuta búsqueda exhaustiva con validación cruzada K-Fold optimizando neg_root_mean_squared_error."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=kf,
        scoring={
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        },
        refit='rmse',
        return_train_score=True,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model_on_test(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """Evalúa un estimador ajustado sobre el conjunto de prueba independiente."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'y_pred': y_pred
    }


def compare_models_dataframe(
    models_summary: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Construye un DataFrame consolidado con métricas de Train, CV y Test para comparar modelos."""
    records = []
    for m in models_summary:
        records.append({
            'Modelo': m['name'],
            'Tipo': m.get('type', 'Optimizado'),
            'CV RMSE (INR)': f"{m['cv_rmse_mean']:.2f} ± {m['cv_rmse_std']:.2f}",
            'CV MAE (INR)': f"{m['cv_mae_mean']:.2f} ± {m['cv_mae_std']:.2f}",
            'CV R²': f"{m['cv_r2_mean']:.4f}",
            'Test RMSE (INR)': f"{m['test_rmse']:.2f}",
            'Test MAE (INR)': f"{m['test_mae']:.2f}",
            'Test R²': f"{m['test_r2']:.4f}",
            'Mejora Test RMSE (%)': f"{m.get('improvement_pct', 0.0):+.2f}%"
        })
    return pd.DataFrame(records)


def save_model_artifact(model: BaseEstimator, filepath: str) -> str:
    """Serializa un pipeline o modelo entrenado a disco usando joblib."""
    import joblib
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    return filepath


def load_model_artifact(filepath: str) -> BaseEstimator:
    """Carga un pipeline o modelo previamente serializado con joblib."""
    import joblib
    return joblib.load(filepath)

