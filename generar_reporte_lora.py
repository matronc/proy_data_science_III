"""Genera el reporte técnico en PDF de la Pre-entrega 4 (Transformer + LoRA).

Lee `resultados_lora.json` (evidencia guardada por el notebook
`notebooks/pre_entrega_4_lora.ipynb`) y arma un PDF de una página por sección
usando únicamente matplotlib, sin dependencias externas de generación de PDF.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
EVIDENCE_PATH = ROOT / "resultados_lora.json"
OUTPUT_PATH = ROOT / "Troncoso_Marcelo_Checkpoint_NLP3.pdf"
AUTOR = "Marcelo Troncoso"


def _new_page(pdf: PdfPages, title: str):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 vertical
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    return fig


def _add_text(fig, text: str, y: float = 0.90, fontsize: int = 10.5):
    fig.text(0.08, y, text, va="top", ha="left", fontsize=fontsize, wrap=True,
              family="DejaVu Sans")


def portada(pdf: PdfPages, evidence: dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.65, "Checkpoint NLP III", ha="center", fontsize=22, fontweight="bold")
    fig.text(0.5, 0.58, "Clasificación con Fine-Tuning Eficiente mediante LoRA",
              ha="center", fontsize=15)
    fig.text(0.5, 0.48, f"Autor: {AUTOR}", ha="center", fontsize=12)
    fig.text(0.5, 0.44, "Dataset: AG News (clasificación de tópicos, 4 clases)",
              ha="center", fontsize=12)
    fig.text(0.5, 0.40, f"Modelo base: {evidence['model_name']}", ha="center", fontsize=12)
    fig.text(0.5, 0.10,
              "Estructura: 1. Resumen de arquitectura  2. Configuración PEFT  "
              "3. Resultados y comparativa  4. Conclusiones",
              ha="center", fontsize=9, style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def seccion_arquitectura(pdf: PdfPages, evidence: dict):
    fig = _new_page(pdf, "1. Resumen de arquitectura")
    texto = (
        f"Modelo base: {evidence['model_name']}\n\n"
        "Justificación: el corpus AG News está en inglés y normalizado a minúsculas, "
        "por lo que no se requiere un modelo multilingüe. DistilBERT retiene ~97% del "
        "rendimiento de BERT-base con ~40% menos parámetros y mayor velocidad de "
        "inferencia/entrenamiento, lo cual es clave dado el hardware disponible para "
        "este checkpoint (una única GPU de gama media). Modelos más grandes (BERT-base, "
        "RoBERTa-base) habrían incrementado el costo computacional sin una mejora "
        "justificada para una tarea de clasificación de 4 clases con vocabulario muy "
        "diferenciado entre categorías.\n\n"
        f"Parámetros totales del modelo: {evidence['total_parameters']:,}\n"
        f"Parámetros entrenables (adaptadores LoRA): {evidence['trainable_parameters']:,}\n"
        f"Porcentaje entrenable: {evidence['trainable_percentage']:.2f}% "
        "(cumple el criterio de eficiencia paramétrica, < 3%).\n\n"
        f"Tiempo de entrenamiento: {evidence['training_seconds']:.2f} segundos "
        f"({evidence['training_seconds'] / 60:.2f} minutos) para 3 épocas."
    )
    _add_text(fig, texto)
    pdf.savefig(fig)
    plt.close(fig)


def seccion_peft(pdf: PdfPages, evidence: dict):
    fig = _new_page(pdf, "2. Configuración PEFT (LoRA)")
    lora = evidence["lora"]
    texto = (
        f"Rango (r): {lora['r']}\n"
        f"Factor de escala (lora_alpha): {lora['alpha']} (razón alpha/r = "
        f"{lora['alpha'] / lora['r']:.1f})\n"
        f"Dropout: {lora['dropout']}\n"
        f"Módulos objetivo: {', '.join(lora['target_modules'])}\n"
        "Bias: none\n\n"
        "Argumentación: la clasificación de tópicos de AG News es una tarea de "
        "complejidad baja-media (4 clases con vocabulario bien diferenciado), por lo "
        "que un rango r=8 alcanza para capturar los patrones específicos de la tarea "
        "sin sobreajustar ni añadir parámetros innecesarios. Un lora_alpha=16 escala "
        "las actualizaciones lo suficiente para que el adaptador tenga impacto real "
        "durante el entrenamiento, manteniendo la estabilidad numérica. El dropout de "
        "0.1 regulariza el adaptador dado el tamaño moderado del dataset (8.000 "
        "ejemplos de entrenamiento). Los módulos q_lin y v_lin corresponden a las "
        "proyecciones de consulta y valor de la atención, el punto estándar de "
        "inserción de adaptadores LoRA según el paper original, ya que concentran la "
        "mayor parte de la capacidad discriminativa del mecanismo de atención con el "
        "menor número de parámetros entrenables.\n\n"
        f"Pérdida final de entrenamiento (train_loss): {evidence['final_train_loss']:.4f}\n"
        f"Pérdida de validación (eval_loss): {evidence['validation_metrics']['eval_loss']:.4f}\n\n"
        "Métricas de validación (mejor checkpoint, época seleccionada por F1 macro):\n"
        f"  Accuracy: {evidence['validation_metrics']['eval_accuracy']:.4f}\n"
        f"  Precision macro: {evidence['validation_metrics']['eval_precision_macro']:.4f}\n"
        f"  Recall macro: {evidence['validation_metrics']['eval_recall_macro']:.4f}\n"
        f"  F1 macro: {evidence['validation_metrics']['eval_f1_macro']:.4f}"
    )
    _add_text(fig, texto)
    pdf.savefig(fig)
    plt.close(fig)


def seccion_resultados(pdf: PdfPages, evidence: dict):
    fig = _new_page(pdf, "3. Resultados y comparativa (conjunto de test)")

    comparison = evidence["comparison"]
    columnas = ["Modelo", "Precisión (macro)", "Recall (macro)", "F1 (macro)"]
    filas = [
        [row["modelo"], f"{row['precision_macro']:.4f}", f"{row['recall_macro']:.4f}",
         f"{row['f1_macro']:.4f}"]
        for row in comparison
    ]

    ax_table = fig.add_axes([0.06, 0.68, 0.90, 0.16])
    ax_table.axis("off")
    tabla = ax_table.table(cellText=filas, colLabels=columnas, loc="center",
                             cellLoc="center", colWidths=[0.42, 0.20, 0.20, 0.18])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9.5)
    tabla.scale(1, 1.8)

    texto = (
        "Reporte de clasificación - DistilBERT + LoRA (test):\n"
        f"{evidence['lora_classification_report']}\n"
        "Reporte de clasificación - TF-IDF + LogisticRegression (test):\n"
        f"{evidence['baseline_classification_report']}"
    )
    fig.text(0.08, 0.62, texto, va="top", ha="left", fontsize=8.3, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def seccion_conclusiones(pdf: PdfPages, evidence: dict):
    fig = _new_page(pdf, "4. Conclusiones")
    baseline_f1 = evidence["baseline_test_metrics"]["f1_macro"]
    lora_f1 = evidence["test_metrics"]["f1_macro"]
    texto = (
        f"F1 macro TF-IDF + LogisticRegression: {baseline_f1:.4f}\n"
        f"F1 macro DistilBERT + LoRA: {lora_f1:.4f}\n"
        f"Diferencia: {lora_f1 - baseline_f1:+.4f}\n\n"
        "En esta ejecución, el baseline TF-IDF + LogisticRegression superó levemente "
        "a DistilBERT + LoRA en el conjunto de test. Para una tarea de clasificación "
        "de tópicos con vocabulario muy distintivo entre clases como AG News, la "
        "representación TF-IDF ya captura casi toda la señal discriminativa, por lo "
        "que la capacidad contextual adicional de un Transformer no se traduce en una "
        "mejora de métricas.\n\n"
        "Sin embargo, el enfoque LoRA demuestra su valor real en eficiencia: solo se "
        f"entrena {evidence['trainable_percentage']:.2f}% de los parámetros del modelo "
        f"({evidence['trainable_parameters']:,} de {evidence['total_parameters']:,}), lo "
        "que permite adaptar un Transformer preentrenado en pocos minutos de GPU sin "
        "actualizar todos sus pesos.\n\n"
        "Conclusión técnica: en este caso concreto, el costo computacional adicional "
        "del fine-tuning no se justifica frente al modelo clásico por la mejora de "
        "métricas obtenida (de hecho, el clásico rindió mejor). El valor de LoRA se "
        "justificaría mejor en tareas de mayor complejidad semántica (matices, "
        "ambigüedad, dominios donde el vocabulario superficial no alcanza), donde la "
        "capacidad contextual del Transformer sí aportaría una ventaja medible frente "
        "al costo adicional de entrenamiento e inferencia."
    )
    _add_text(fig, texto, fontsize=10.5)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    with PdfPages(OUTPUT_PATH) as pdf:
        portada(pdf, evidence)
        seccion_arquitectura(pdf, evidence)
        seccion_peft(pdf, evidence)
        seccion_resultados(pdf, evidence)
        seccion_conclusiones(pdf, evidence)
    print(f"Reporte generado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
