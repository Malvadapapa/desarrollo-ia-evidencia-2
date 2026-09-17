"""Módulo de entrenamiento, inferencia y evaluación de modelos de regresión lineal.
Diseñado para la Evidencia de Aprendizaje N.º 2 - Desarrollo de IA (ISPC).
"""

import os
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
