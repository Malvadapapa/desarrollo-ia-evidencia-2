"""Módulo de inferencia y puesta en producción para predicción del costo unitario de neumáticos.
Diseñado para la Evidencia de Aprendizaje N.º 3 - Desarrollo de IA (ISPC).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_optimization import load_model_artifact

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "best_model.joblib"
)


def predict_single_tire(
    rim: float,
    width: float,
    ratio: Optional[float],
    brand: str,
    model_path: str = DEFAULT_MODEL_PATH
) -> float:
    """Predice el costo unitario de un neumático individual dadas sus dimensiones y marca.
    
    Parámetros:
    -----------
    rim : float
        Diámetro del rin en pulgadas (ej. 15.0, 17.0, 22.5).
    width : float
        Ancho de sección en milímetros (ej. 195.0, 205.0, 279.4).
    ratio : float or None
        Perfil o relación de aspecto en porcentaje (ej. 65.0, 55.0). Si es None o np.nan,
        el pipeline imputa automáticamente la mediana aprendida durante el entrenamiento.
    brand : str
        Nombre del fabricante (ej. 'BRAND_A', 'BRAND_B', 'BRAND_C', 'BRAND_D').
    model_path : str
        Ruta al archivo serializado (.joblib) del modelo entrenado.
        
    Retorna:
    --------
    float: Costo unitario predicho en INR.
    """
    model = load_model_artifact(model_path)
    input_df = pd.DataFrame([{
        'rim': float(rim),
        'width': float(width),
        'ratio': np.nan if ratio is None else float(ratio),
        'Brand': str(brand).strip()
    }])
    pred = model.predict(input_df)[0]
    return float(pred)


def predict_batch_from_csv(
    input_csv: str,
    output_csv: str,
    model_path: str = DEFAULT_MODEL_PATH
) -> pd.DataFrame:
    """Genera predicciones por lote a partir de un archivo CSV y guarda el resultado."""
    model = load_model_artifact(model_path)
    df = pd.read_csv(input_csv)
    
    required_cols = ['rim', 'width', 'ratio', 'Brand']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"El archivo CSV debe contener la columna obligatoria '{col}'")
            
    df['predicted_unit_cost_inr'] = model.predict(df[required_cols])
    df.to_csv(output_csv, index=False)
    return df


def main():
    """Punto de entrada de línea de comandos para inferencia de costo."""
    parser = argparse.ArgumentParser(
        description="Inferencia de Costo Unitario de Neumáticos - Desarrollo de IA (ISPC)"
    )
    parser.add_argument("--rim", type=float, help="Diámetro del rin en pulgadas (ej. 16.0)")
    parser.add_argument("--width", type=float, help="Ancho de sección en mm (ej. 205.0)")
    parser.add_argument("--ratio", type=float, default=None, help="Perfil / Aspect Ratio en %% (opcional, ej. 55.0)")
    parser.add_argument("--brand", type=str, help="Marca del fabricante (ej. BRAND_A, BRAND_B, BRAND_C, BRAND_D)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Ruta al modelo joblib")
    parser.add_argument("--input-csv", type=str, default=None, help="Ruta a CSV de entrada para inferencia batch")
    parser.add_argument("--output-csv", type=str, default=None, help="Ruta a CSV de salida para inferencia batch")

    args = parser.parse_args()

    if args.input_csv:
        if not args.output_csv:
            sys.exit("Error: Se requiere --output-csv al usar --input-csv")
        print(f"Procesando inferencia batch desde: {args.input_csv}")
        df_out = predict_batch_from_csv(args.input_csv, args.output_csv, model_path=args.model)
        print(f"Predicciones guardadas exitosamente en: {args.output_csv} ({len(df_out)} registros)")
        return

    if args.rim is None or args.width is None or args.brand is None:
        parser.print_help()
        sys.exit("\nError: Debe especificar --rim, --width y --brand (o --input-csv para inferencia batch).")

    pred_cost = predict_single_tire(
        rim=args.rim,
        width=args.width,
        ratio=args.ratio,
        brand=args.brand,
        model_path=args.model
    )
    print("=" * 60)
    print("PREDICCIÓN DE COSTO UNITARIO - NEUMÁTICOS")
    print(f"  Rin: {args.rim:.1f} pulgadas | Ancho: {args.width:.1f} mm | Ratio: {args.ratio if args.ratio else 'Imputado'} | Marca: {args.brand}")
    print(f"  Costo Unitario Estimado: ${pred_cost:,.2f} INR")
    print("=" * 60)


if __name__ == "__main__":
    main()
