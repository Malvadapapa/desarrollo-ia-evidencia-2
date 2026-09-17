"""Suite de pruebas automatizadas de integridad, invariantes y reproducibilidad.
Diseñado para la Evidencia de Aprendizaje N.º 2 - Desarrollo de IA (ISPC).
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    clean_unit_cost,
    parse_tire_specs,
    load_and_preprocess_data,
    split_data,
    impute_aspect_ratio,
    encode_brands
)
from src.regression_models import (
    train_simple_model,
    train_multiple_model,
    evaluate_model,
    plot_simple_regression,
    plot_real_vs_pred,
    plot_residuals
)


class TestTireParser(unittest.TestCase):
    """Pruebas unitarias para validar la extracción física del parser y el manejo de faltantes."""

    def test_metric_explicit_ratio(self):
        w, r, rim = parse_tire_specs("145/70 R 12 BY-801")
        self.assertEqual(w, 145.0)
        self.assertEqual(r, 70.0)
        self.assertEqual(rim, 12.0)

    def test_metric_with_zr(self):
        w, r, rim = parse_tire_specs("225/40 Z R 19 TR41")
        self.assertEqual(w, 225.0)
        self.assertEqual(r, 40.0)
        self.assertEqual(rim, 19.0)

    def test_flotation_geometric_deduction(self):
        # 31 x 10.50 R 15: H = (31 - 15) / 2 = 8; Ratio = (8 / 10.5) * 100 = 76.1904...
        w, r, rim = parse_tire_specs("31 x 10.50 R 15 LT / 6PR BY-AT987")
        self.assertAlmostEqual(w, 10.50 * 25.4, places=2)
        self.assertEqual(rim, 15.0)
        self.assertAlmostEqual(r, (8.0 / 10.50) * 100.0, places=2)

    def test_commercial_c_missing_ratio(self):
        # Casos comerciales sin barra: el parser NO debe inventar un número, debe devolver np.nan
        w, r, rim = parse_tire_specs("155 R 12 C / 8PR BY-05")
        self.assertEqual(w, 155.0)
        self.assertEqual(rim, 12.0)
        self.assertTrue(np.isnan(r), "El parser no debe inventar el Aspect Ratio en comerciales sin barra")

    def test_heavy_truck_special_cases(self):
        # 1000 R 20: 10.00 pulgadas = 254 mm
        w1, r1, rim1 = parse_tire_specs("1000 R 20/18 PR LO105 (SET)")
        self.assertEqual(w1, 254.0)
        self.assertEqual(rim1, 20.0)
        self.assertTrue(np.isnan(r1))

        # 11 R 22.5: 11.00 pulgadas = 279.4 mm
        w2, r2, rim2 = parse_tire_specs("11 R 22.5/ 18 PR (LO301)")
        self.assertEqual(w2, 279.4)
        self.assertEqual(rim2, 22.5)
        self.assertTrue(np.isnan(r2))


class TestPipelineAndModels(unittest.TestCase):
    """Pruebas de integración para el pipeline de preprocesamiento, modelado e invariantes."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/DS_Data_Set_Current_Stock.csv"
        cls.df = load_and_preprocess_data(cls.data_path)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_data(cls.df, test_size=0.20, random_state=42)
        cls.X_train_imp, cls.X_test_imp, cls.imputed_val = impute_aspect_ratio(cls.X_train, cls.X_test)
        cls.X_train_enc, cls.X_test_enc = encode_brands(cls.X_train_imp, cls.X_test_imp, reference_brand='BRAND_A')

    def test_dataset_cleaning_and_dimensions(self):
        # 253 originales - 2 nulos = 251 observaciones supervisadas
        self.assertEqual(len(self.df), 251)
        self.assertEqual(len(self.X_train), 200)
        self.assertEqual(len(self.X_test), 51)
        self.assertEqual(len(self.y_train), 200)
        self.assertEqual(len(self.y_test), 51)

    def test_no_nans_after_preprocessing(self):
        self.assertEqual(self.X_train_enc.isna().sum().sum(), 0)
        self.assertEqual(self.X_test_enc.isna().sum().sum(), 0)
        self.assertEqual(self.y_train.isna().sum(), 0)
        self.assertEqual(self.y_test.isna().sum(), 0)

    def test_brand_reference_and_columns_alignment(self):
        expected_cols = ['rim', 'width', 'ratio', 'Brand_BRAND_B', 'Brand_BRAND_C', 'Brand_BRAND_D']
        self.assertListEqual(list(self.X_train_enc.columns), expected_cols)
        self.assertListEqual(list(self.X_test_enc.columns), expected_cols)
        self.assertNotIn('Brand_BRAND_A', self.X_train_enc.columns, "BRAND_A debe ser la base omitida")

    def test_models_fitting_and_finite_metrics(self):
        m_simple = train_simple_model(self.X_train_enc, self.y_train)
        m_mult = train_multiple_model(self.X_train_enc, self.y_train)

        # Finitud de coeficientes
        self.assertTrue(np.isfinite(m_simple.intercept_))
        self.assertTrue(all(np.isfinite(m_simple.coef_)))
        self.assertTrue(np.isfinite(m_mult.intercept_))
        self.assertTrue(all(np.isfinite(m_mult.coef_)))

        # Evaluación sobre test
        res_s = evaluate_model(m_simple, self.X_test_enc[['rim']], self.y_test)
        res_m = evaluate_model(m_mult, self.X_test_enc, self.y_test)

        self.assertGreater(res_s['mae'], 0)
        self.assertGreater(res_s['rmse'], 0)
        self.assertLessEqual(res_s['r2'], 1.0)

        self.assertGreater(res_m['mae'], 0)
        self.assertGreater(res_m['rmse'], 0)
        self.assertLessEqual(res_m['r2'], 1.0)

        self.assertEqual(len(res_s['y_pred']), 51)
        self.assertEqual(len(res_m['y_pred']), 51)

    def test_reproducibility(self):
        # Repetir el split y ajuste con random_state=42
        X_tr2, X_te2, y_tr2, y_te2 = split_data(self.df, test_size=0.20, random_state=42)
        X_tr2_imp, X_te2_imp, _ = impute_aspect_ratio(X_tr2, X_te2)
        X_tr2_enc, X_te2_enc = encode_brands(X_tr2_imp, X_te2_imp, reference_brand='BRAND_A')

        m2 = train_multiple_model(X_tr2_enc, y_tr2)
        m1 = train_multiple_model(self.X_train_enc, self.y_train)

        diff = np.max(np.abs(m1.coef_ - m2.coef_))
        self.assertEqual(diff, 0.0, "El modelo debe ser 100% reproducible con la misma semilla")

    def test_plot_artifacts_generation(self):
        m_simple = train_simple_model(self.X_train_enc, self.y_train)
        m_mult = train_multiple_model(self.X_train_enc, self.y_train)
        res_s = evaluate_model(m_simple, self.X_test_enc[['rim']], self.y_test)
        res_m = evaluate_model(m_mult, self.X_test_enc, self.y_test)

        p1 = "reports/figures/test_plot_simple.png"
        p2 = "reports/figures/test_plot_real_vs_pred.png"
        p3 = "reports/figures/test_plot_residuals.png"

        plot_simple_regression(m_simple, self.X_test_enc, self.y_test, output_path=p1)
        plot_real_vs_pred(self.y_test, res_s['y_pred'], res_m['y_pred'], output_path=p2)
        plot_residuals(self.y_test, res_m['y_pred'], output_path=p3)

        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))
        self.assertTrue(os.path.exists(p3))

        # Limpiar figuras de test
        for p in [p1, p2, p3]:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    unittest.main()
