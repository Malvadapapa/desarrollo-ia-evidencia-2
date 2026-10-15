"""Suite de pruebas unitarias y de integración para la optimización de modelos y pipelines.
Diseñado para la Evidencia de Aprendizaje N.º 3 - Desarrollo de IA (ISPC).
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.model_optimization import (
    build_preprocessor,
    build_pipeline,
    get_feature_names,
    cross_validate_pipeline,
    run_grid_search,
    evaluate_model_on_test,
    compare_models_dataframe
)
from src.regression_models import (
    plot_cv_comparison,
    plot_feature_importances,
    plot_actual_vs_predicted_advanced,
    plot_residuals_diagnostics_advanced
)


class TestOptimizationPipeline(unittest.TestCase):
    """Pruebas del preprocesador ColumnTransformer y pipelines integrados."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/DS_Data_Set_Current_Stock.csv"
        cls.df = load_and_preprocess_data(cls.data_path)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_data(
            cls.df, test_size=0.20, random_state=42
        )

    def test_preprocessor_structure_and_no_leakage(self):
        """Verifica que el ColumnTransformer no deje nulos y excluya la marca de referencia."""
        pre = build_preprocessor(with_scaler=False, reference_brand='BRAND_A')
        X_trans = pre.fit_transform(self.X_train)
        
        self.assertEqual(np.isnan(X_trans).sum(), 0, "No deben existir valores NaN tras la transformación")
        feature_names = get_feature_names(pre)
        
        expected_features = ['rim', 'width', 'ratio', 'Brand_BRAND_B', 'Brand_BRAND_C', 'Brand_BRAND_D']
        self.assertListEqual(feature_names, expected_features)
        self.assertNotIn('Brand_BRAND_A', feature_names, "BRAND_A debe ser la base omitida")

    def test_pipeline_reproduces_evidencia_2_baseline(self):
        """Verifica que el Pipeline reproduzca exactamente las métricas de la Evidencia 2."""
        pipe_ols = build_pipeline(LinearRegression(), with_scaler=False)
        pipe_ols.fit(self.X_train, self.y_train)
        
        metrics = evaluate_model_on_test(pipe_ols, self.X_test, self.y_test)
        
        # En Evidencia 2: MAE=1678.60, RMSE=2549.50, R²=0.7968
        self.assertAlmostEqual(metrics['mae'], 1678.60, delta=0.1)
        self.assertAlmostEqual(metrics['rmse'], 2549.50, delta=0.1)
        self.assertAlmostEqual(metrics['r2'], 0.7968, delta=0.001)

    def test_cross_validation_execution(self):
        """Verifica que la validación cruzada 5-Fold devuelva métricas finitas y consistentes."""
        pipe = build_pipeline(LinearRegression(), with_scaler=False)
        res = cross_validate_pipeline(pipe, self.X_train, self.y_train, cv=5, random_state=42)
        
        self.assertEqual(len(res['cv_rmse_folds']), 5)
        self.assertEqual(len(res['cv_r2_folds']), 5)
        self.assertTrue(np.all(np.isfinite(res['cv_rmse_folds'])))
        self.assertGreater(res['cv_rmse_mean'], 0)
        self.assertLessEqual(res['cv_r2_mean'], 1.0)
        self.assertGreater(res['train_r2_mean'], res['cv_r2_mean'], "Train R² suele ser mayor al CV R²")

    def test_grid_search_mini_grid(self):
        """Verifica el funcionamiento de GridSearchCV con una grilla representativa."""
        pipe = build_pipeline(Ridge(), with_scaler=True)
        param_grid = {'regressor__alpha': [1.0, 10.0]}
        
        gs = run_grid_search(pipe, param_grid, self.X_train, self.y_train, cv=3, random_state=42)
        
        self.assertIn('regressor__alpha', gs.best_params_)
        self.assertTrue(hasattr(gs, 'best_estimator_'))
        
        preds = gs.predict(self.X_test)
        self.assertEqual(len(preds), len(self.y_test))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_models_comparison_dataframe_builder(self):
        """Verifica que la tabla consolidada de comparación se genere adecuadamente."""
        summary = [
            {
                'name': 'OLS Baseline',
                'type': 'Lineal OLS',
                'cv_rmse_mean': 2589.29,
                'cv_rmse_std': 415.12,
                'cv_mae_mean': 1850.10,
                'cv_mae_std': 210.00,
                'cv_r2_mean': 0.7575,
                'test_rmse': 2549.50,
                'test_mae': 1678.60,
                'test_r2': 0.7968,
                'improvement_pct': 0.0
            },
            {
                'name': 'Gradient Boosting',
                'type': 'Ensamble Boosting',
                'cv_rmse_mean': 1432.78,
                'cv_rmse_std': 280.45,
                'cv_mae_mean': 1020.15,
                'cv_mae_std': 145.20,
                'cv_r2_mean': 0.8850,
                'test_rmse': 1577.13,
                'test_mae': 975.85,
                'test_r2': 0.9223,
                'improvement_pct': 38.14
            }
        ]
        df_comp = compare_models_dataframe(summary)
        self.assertEqual(len(df_comp), 2)
        self.assertIn('CV RMSE (INR)', df_comp.columns)
        self.assertIn('Test R²', df_comp.columns)

    def test_diagnostic_plots_generation(self):
        """Verifica la generación de figuras diagnósticas de la Evidencia 3."""
        p_cv = "reports/figures/test_ev3_cv_comp.png"
        p_feat = "reports/figures/test_ev3_feat_imp.png"
        p_scatter = "reports/figures/test_ev3_actual_vs_pred.png"
        p_res = "reports/figures/test_ev3_residuals.png"

        dummy_cv = {
            'OLS': np.array([2400.0, 2500.0, 2600.0, 2550.0, 2700.0]),
            'Random Forest': np.array([1400.0, 1350.0, 1420.0, 1390.0, 1360.0])
        }
        plot_cv_comparison(dummy_cv, output_path=p_cv)
        
        features = ['rim', 'width', 'ratio', 'Brand_B', 'Brand_C', 'Brand_D']
        importances = np.array([0.45, 0.20, 0.15, 0.05, 0.05, 0.10])
        plot_feature_importances(features, importances, model_name='RF Test', output_path=p_feat)
        
        y_dummy = pd.Series([5000.0, 10000.0, 15000.0, 20000.0])
        preds_dummy = {
            'OLS (Baseline Ev2)': np.array([5200.0, 9500.0, 14800.0, 19200.0]),
            'Gradient Boosting (Ganador)': np.array([5050.0, 9900.0, 15100.0, 20050.0])
        }
        plot_actual_vs_predicted_advanced(y_dummy, preds_dummy, output_path=p_scatter)
        plot_residuals_diagnostics_advanced(y_dummy, preds_dummy, output_path=p_res)
        
        for path in [p_cv, p_feat, p_scatter, p_res]:
            self.assertTrue(os.path.exists(path), f"El archivo {path} debe existir")
            os.remove(path)

    def test_model_export_and_inference(self):
        """Verifica la persistencia con joblib y la inferencia individual."""
        from src.model_optimization import save_model_artifact, load_model_artifact
        from src.predict import predict_single_tire
        
        pipe = build_pipeline(LinearRegression())
        pipe.fit(self.X_train, self.y_train)
        
        tmp_model_path = "models/test_temp_model.joblib"
        save_model_artifact(pipe, tmp_model_path)
        self.assertTrue(os.path.exists(tmp_model_path))
        
        reloaded = load_model_artifact(tmp_model_path)
        self.assertTrue(hasattr(reloaded, 'predict'))
        
        pred = predict_single_tire(rim=15.0, width=195.0, ratio=65.0, brand='BRAND_B', model_path=tmp_model_path)
        self.assertTrue(np.isfinite(pred))
        self.assertGreater(pred, 0)
        
        os.remove(tmp_model_path)


if __name__ == "__main__":
    unittest.main()

