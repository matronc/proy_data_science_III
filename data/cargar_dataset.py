"""
Loader de los datasets de la cátedra (Data Science III: NLP & Deep Learning).
Sirve para las pre-entregas de los Módulos 2, 3, 4 y el Proyecto Final.

Elegí UN dataset al inicio del Módulo 2 y usá el MISMO hasta el final.
Los splits train/test ya vienen separados: NUNCA entrenes con el test.
"""

import pandas as pd
from pathlib import Path

# Cambiá a "bbc_news" si preferís artículos largos.
DATASET = "ag_news"            # opciones: "ag_news" | "bbc_news"
BASE = Path(__file__).parent   # carpeta donde está este script


def cargar(dataset: str = DATASET):
    """Devuelve (train_df, test_df) con columnas: text, label."""
    dataset_dir = BASE / dataset
    if not dataset_dir.exists():
        dataset_dir = BASE

    tr = pd.read_csv(dataset_dir / f"{dataset}_train.csv")
    te = pd.read_csv(dataset_dir / f"{dataset}_test.csv")
    return tr, te


if __name__ == "__main__":
    train_df, test_df = cargar()
    print(f"Dataset: {DATASET}")
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")
    print("Clases:", sorted(train_df['label'].unique()))
    print(train_df.head())
