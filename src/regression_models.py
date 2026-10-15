"""Módulo de entrenamiento, inferencia y evaluación de modelos de regresión lineal y ensambles.
Diseñado para las Evidencias de Aprendizaje N.º 2 y N.º 3 - Desarrollo de IA (ISPC).
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_simple_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Ajusta un modelo de Regresión Lineal Simple utilizando únicamente 'rim'."""
    model = LinearRegression()
    model.fit(X_train[['rim']], y_train)
    return model


def train_multiple_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Ajusta un modelo de Regresión Lineal Múltiple utilizando todas las variables preparadas."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcula las métricas formales de regresión: MAE, MSE, RMSE y R²."""
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'y_pred': y_pred
    }


def plot_simple_regression(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series, output_path: str = None):
    """Genera el gráfico de dispersión y recta de regresión simple."""
    plt.figure(figsize=(8, 5))
    
    # Ordenar para trazar la línea continua
    X_plot = pd.DataFrame({'rim': np.linspace(X_test['rim'].min(), X_test['rim'].max(), 100)})
    y_plot = model.predict(X_plot)
    
    plt.scatter(X_test['rim'], y_test, color='#1f77b4', alpha=0.7, edgecolors='k', label='Datos reales (Test)')
    plt.plot(X_plot['rim'], y_plot, color='#d62728', linewidth=2, label=f'Recta estimada (β₁ = {model.coef_[0]:.2f})')
    
    plt.title('Regresión Lineal Simple: Diámetro del Rin vs Costo Unitario', fontsize=12, pad=10)
    plt.xlabel('Diámetro del Rin (Pulgadas)', fontsize=10)
    plt.ylabel('Costo Unitario (INR)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_real_vs_pred(y_test: pd.Series, y_pred_simple: np.ndarray, y_pred_mult: np.ndarray, output_path: str = None):
    """Genera el gráfico de Valores Reales vs Predichos con línea identidad y=x."""
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test, y_pred_simple, color='#ff7f0e', alpha=0.6, label='Modelo Simple (Rim)')
    plt.scatter(y_test, y_pred_mult, color='#2ca02c', alpha=0.7, edgecolors='k', label='Modelo Múltiple (Specs + Brand)')
    
    # Línea de referencia ideal y = x
    min_val = min(y_test.min(), y_pred_mult.min())
    max_val = max(y_test.max(), y_pred_mult.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Predicción perfecta (y = x)')
    
    plt.title('Comparación: Valores Reales vs. Valores Predichos (Conjunto Test)', fontsize=12, pad=10)
    plt.xlabel('Costo Unitario Real (INR)', fontsize=10)
    plt.ylabel('Costo Unitario Predicho (INR)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residuals(y_test: pd.Series, y_pred_mult: np.ndarray, output_path: str = None):
    """Genera el gráfico de Residuos vs Valores Predichos para inspección diagnóstica."""
    residuals = y_test - y_pred_mult
    
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_mult, residuals, color='#9467bd', alpha=0.7, edgecolors='k')
    plt.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Residuo cero')
    
    plt.title('Diagnóstico de Residuos: Modelo Múltiple (Conjunto Test)', fontsize=12, pad=10)
    plt.xlabel('Costo Unitario Predicho (INR)', fontsize=10)
    plt.ylabel('Residuo (Real - Predicho) en INR', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


# =========================================================================
# Nuevas visualizaciones diagnósticas para Evidencia 3 (Optimización y Comparación)
# =========================================================================

def plot_cv_comparison(cv_scores_dict: Dict[str, np.ndarray], output_path: Optional[str] = None):
    """Genera un boxplot de comparación de RMSE en Validación Cruzada (K=5) entre modelos."""
    plt.figure(figsize=(9, 5))
    
    models = list(cv_scores_dict.keys())
    data = [cv_scores_dict[m] for m in models]
    
    box = plt.boxplot(data, tick_labels=models, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    
    colors = ['#aec7e8', '#c7c7c7', '#dbdb8d', '#98df8a', '#2ca02c']
    for patch, color in zip(box['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        
    plt.title('Comparación de Modelos: Distribución de RMSE en Validación Cruzada (5-Fold)', fontsize=12, pad=12)
    plt.ylabel('RMSE en Validación Cruzada (INR)', fontsize=10)
    plt.xlabel('Algoritmo / Modelo', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = 'Gradient Boosting',
    output_path: Optional[str] = None
):
    """Genera un gráfico de barras horizontales con la importancia de características."""
    plt.figure(figsize=(8, 5))
    
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    bars = plt.barh(range(len(indices)), sorted_importances, color='#1f77b4', alpha=0.85, edgecolor='black')
    plt.yticks(range(len(indices)), sorted_features, fontsize=9)
    plt.xlabel('Importancia Relativa (Gini / MDI / Impureza)', fontsize=10)
    plt.title(f'Importancia de Variables en el Modelo Ganador ({model_name})', fontsize=12, pad=12)
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    # Anotación del valor sobre cada barra
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f'{w*100:.1f}%', va='center', fontsize=9)
        
    plt.xlim(0, max(sorted_importances) * 1.15)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted_advanced(
    y_test: pd.Series,
    predictions_dict: Dict[str, np.ndarray],
    output_path: Optional[str] = None
):
    """Genera una comparación gráfica de Valores Reales vs Predichos para múltiples modelos en Test."""
    plt.figure(figsize=(9, 6))
    
    colors = {
        'OLS (Baseline Ev2)': '#ff7f0e',
        'Ridge': '#17becf',
        'Random Forest': '#9467bd',
        'Gradient Boosting (Ganador)': '#2ca02c'
    }
    
    min_val = min(y_test.min(), min(p.min() for p in predictions_dict.values()))
    max_val = max(y_test.max(), max(p.max() for p in predictions_dict.values()))
    
    # Línea ideal y = x
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Predicción perfecta (y = x)')
    
    for name, y_pred in predictions_dict.items():
        color = colors.get(name, '#7f7f7f')
        alpha = 0.8 if 'Ganador' in name else 0.5
        size = 45 if 'Ganador' in name else 30
        plt.scatter(y_test, y_pred, label=name, color=color, alpha=alpha, s=size, edgecolors='none' if alpha < 0.7 else 'k')
        
    plt.title('Comparación en Test (N=51): Valores Reales vs. Predichos', fontsize=12, pad=12)
    plt.xlabel('Costo Unitario Real (INR)', fontsize=10)
    plt.ylabel('Costo Unitario Predicho (INR)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residuals_diagnostics_advanced(
    y_test: pd.Series,
    preds_dict: Dict[str, np.ndarray],
    output_path: Optional[str] = None
):
    """Genera diagnóstico comparativo de residuos (dispersión y densidad) para Baseline vs Ganador."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    colors = {'OLS (Baseline Ev2)': '#ff7f0e', 'Gradient Boosting (Ganador)': '#2ca02c', 'Random Forest': '#9467bd'}
    
    # Panel 1: Residuos vs Predicción
    for name, y_pred in preds_dict.items():
        residuals = y_test - y_pred
        color = colors.get(name, '#1f77b4')
        axes[0].scatter(y_pred, residuals, label=name, color=color, alpha=0.6, edgecolors='k' if 'Ganador' in name else 'none')
        
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Residuos vs. Valores Predichos', fontsize=11)
    axes[0].set_xlabel('Costo Unitario Predicho (INR)', fontsize=10)
    axes[0].set_ylabel('Residuo (Real - Predicho) [INR]', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True)
    
    # Panel 2: Histograma de Residuos
    for name, y_pred in preds_dict.items():
        residuals = y_test - y_pred
        color = colors.get(name, '#1f77b4')
        axes[1].hist(residuals, bins=15, alpha=0.5, label=name, color=color, edgecolor='black', density=True)
        
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Distribución de Densidad de Residuos', fontsize=11)
    axes[1].set_xlabel('Residuo (INR)', fontsize=10)
    axes[1].set_ylabel('Densidad', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(frameon=True)
    
    plt.suptitle('Diagnóstico de Residuos en Test: Modelo Base vs. Modelo Optimizado', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
