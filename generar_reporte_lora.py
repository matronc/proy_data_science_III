"""Genera el informe tecnico consolidado del proyecto NLP en PDF."""
from __future__ import annotations

import html
import json
import re
import textwrap
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
EVIDENCE_PATH = ROOT / "resultados_lora.json"
OUTPUT_PATH = ROOT / "NLP_Capstone_Troncoso_Marcelo.pdf"
AUTHOR = "Marcelo Troncoso"
LABELS = ["Business", "Sci_Tech", "Sports", "World"]


LEFT_MARGIN = 0.09
RIGHT_MARGIN = 0.93


def _page(pdf: PdfPages, title: str):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.965)
    return fig


def _wrap_paragraphs(value: str, width: int) -> str:
    paragraphs = value.split("\n\n")
    wrapped_paragraphs = []
    for paragraph in paragraphs:
        lines = paragraph.split("\n")
        wrapped_lines = [textwrap.fill(line, width=width) if line.strip() else line
                         for line in lines]
        wrapped_paragraphs.append("\n".join(wrapped_lines))
    return "\n\n".join(wrapped_paragraphs)


def _text(fig, value: str, y: float = 0.90, size: float = 10.2):
    width = max(60, int(862 / size))
    fig.text(LEFT_MARGIN, y, _wrap_paragraphs(value, width), va="top", ha="left",
              fontsize=size, family="DejaVu Sans", linespacing=1.35)


def _parse_weighted_avg(report_text: str) -> tuple[float, float, float]:
    match = re.search(
        r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", report_text)
    if not match:
        raise ValueError("No se encontro la fila 'weighted avg' en el reporte.")
    precision, recall, f1 = (float(value) for value in match.groups())
    return precision, recall, f1


def _baseline_confusion():
    train = pd.read_csv(ROOT / "data" / "ag_news_train.csv")
    test = pd.read_csv(ROOT / "data" / "ag_news_test.csv")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    html_re = re.compile(r"<[^>]+>")
    url_re = re.compile(r"https?://\S+|www\.\S+")
    alpha_re = re.compile(r"[^a-zA-Z\s]")
    spaces_re = re.compile(r"\s+")

    def preprocess(value: str) -> str:
        value = html.unescape(str(value))
        value = html_re.sub(" ", value)
        value = url_re.sub(" ", value)
        value = alpha_re.sub(" ", value.lower())
        value = spaces_re.sub(" ", value).strip()
        doc = nlp(value)
        return " ".join(token.lemma_ for token in doc
                        if not token.is_space and not token.is_punct
                        and len(token.lemma_) > 1)

    model = joblib.load(ROOT / "modelo_tfidf_ag_news.joblib")
    train_processed = train["text"].fillna("").map(preprocess)
    predictions = model.predict(test["text"].fillna("").map(preprocess))
    start_time = time.perf_counter()
    clone(model).fit(train_processed, train["label"])
    training_seconds = time.perf_counter() - start_time
    labels = sorted(train["label"].unique())
    matrix = confusion_matrix(test["label"], predictions, labels=labels)
    report_dict = classification_report(test["label"], predictions, labels=labels, output_dict=True)
    weighted = report_dict["weighted avg"]
    weighted_metrics = (weighted["precision"], weighted["recall"], weighted["f1-score"])

    mismatches = test.assign(prediction=predictions)
    mismatches = mismatches[mismatches["label"] != mismatches["prediction"]]
    confusable = mismatches[(mismatches["label"] == "Business") & (mismatches["prediction"] == "Sci_Tech")]
    example_row = confusable.iloc[0] if not confusable.empty else mismatches.iloc[0]
    example = (str(example_row["text"])[:280], example_row["label"], example_row["prediction"])

    return labels, matrix, report_dict, training_seconds, weighted_metrics, example


def _heatmap(fig, matrix, labels, title, position):
    ax = fig.add_axes(position)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Prediccion", fontsize=8)
    ax.set_ylabel("Real", fontsize=8)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center",
                    fontsize=9, color="white" if matrix[row, column] > matrix.max() * .55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _classification_table(fig, report_dict, labels, position):
    rows = []
    for label in labels:
        metrics = report_dict[label]
        rows.append([label, f"{metrics['precision']:.2f}", f"{metrics['recall']:.2f}",
                     f"{metrics['f1-score']:.2f}", f"{int(metrics['support'])}"])
    rows.append(["accuracy", "", "", f"{report_dict['accuracy']:.2f}",
                 f"{int(report_dict['macro avg']['support'])}"])
    for key, display in (("macro avg", "macro avg"), ("weighted avg", "weighted avg")):
        metrics = report_dict[key]
        rows.append([display, f"{metrics['precision']:.2f}", f"{metrics['recall']:.2f}",
                     f"{metrics['f1-score']:.2f}", f"{int(metrics['support'])}"])
    ax = fig.add_axes(position)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Clase", "Precision", "Recall", "F1", "Support"],
                     loc="center", cellLoc="center", colWidths=[0.34, 0.16, 0.16, 0.16, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)


def cover(pdf: PdfPages, evidence: dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.78, "INFORME TECNICO", fontsize=12, color="#1f5f75", fontweight="bold")
    fig.text(0.08, 0.70, "Clasificacion de noticias\ncon NLP clasico y LoRA", fontsize=25,
             fontweight="bold", color="#152b36")
    fig.text(0.08, 0.54, "Proyecto final consolidado | AG News", fontsize=14, color="#1f5f75")
    fig.text(0.08, 0.46, f"Autor: {AUTHOR}\nModelo avanzado: {evidence['model_name']}\n"
             "Comparacion sobre el mismo conjunto de test (2.000 noticias)", fontsize=11,
             linespacing=1.6)
    fig.text(0.08, 0.12, "TF-IDF + LogisticRegression frente a DistilBERT + LoRA\n"
             "Resultado principal: el baseline clasico obtiene el mejor F1 macro en este corpus.",
             fontsize=10, color="#3f5963")
    pdf.savefig(fig)
    plt.close(fig)


def problem_and_pipeline(pdf: PdfPages):
    fig = _page(pdf, "1. Problema, datos y pipeline end-to-end")
    text = (
        "Problema. Clasificar titulares y textos breves de AG News en cuatro temas: "
        "Business, Sci_Tech, Sports y World. El objetivo es establecer un baseline "
        "interpretable y contrastarlo con transfer learning eficiente para elegir una "
        "solucion segun rendimiento, costo y complejidad operativa.\n\n"
        "Datos y protocolo. El corpus contiene 8.000 ejemplos de entrenamiento y 2.000 "
        "ejemplos de test, con 500 observaciones por clase en test. El train se divide "
        "estratificadamente en 6.400 ejemplos de ajuste y 1.600 de validacion. El test "
        "se reserva hasta la evaluacion final y es exactamente el mismo para ambos modelos.\n\n"
        "Pipeline clasico. Se desescapan entidades HTML, se eliminan etiquetas, URLs, "
        "caracteres no alfabeticos y espacios redundantes; luego se normaliza a minusculas, "
        "se tokeniza y lematiza con spaCy en_core_web_sm. Scikit-Learn recibe esa secuencia "
        "lematizada y su TfidfVectorizer construye el vocabulario de unigramas y bigramas; "
        "no utiliza tokens especiales ni embeddings densos. TF-IDF usa stop_words='english', "
        "min_df=2, max_df=0.95, sublinear_tf=True, 20.000 features maximas y unigramas/bigramas. "
        "La regresion logistica se ajusta sobre matrices dispersas.\n\n"
        "Pipeline Transformer. DistilBERT recibe el texto original normalizado a minusculas; "
        "la tokenizacion subword (WordPiece) de Hugging Face agrega los tokens especiales "
        "[CLS] y [SEP], trunca a max_length=128 tokens y realiza padding dinamico por batch "
        "(longitud variable segun el lote, nunca mayor a 128). Cada token se convierte en un "
        "embedding denso contextual; las etiquetas se mapean a enteros y el test no participa "
        "en entrenamiento ni seleccion de hiperparametros."
    )
    _text(fig, text, size=10)
    pdf.savefig(fig)
    plt.close(fig)


def baseline(pdf: PdfPages, labels, matrix, report_dict):
    fig = _page(pdf, "2. Baseline: TF-IDF + LogisticRegression")
    text = (
        "La configuracion seleccionada alcanzo F1 macro de validacion 0.8927 frente a 0.8902 "
        "de unigramas. El modelo final se reajusto sobre los 8.000 ejemplos de train y produjo "
        "una matriz TF-IDF de 20.000 columnas.\n\n"
        "En test, Sports es la clase mas separable (F1=0.96), mientras Business (F1=0.85) y "
        "Sci_Tech (F1=0.87) concentran el desafio. La razon es el vocabulario compartido de "
        "empresas, mercados, tecnologia y resultados financieros. La representacion pondera "
        "terminos y n-gramas discriminativos, pero no modela relaciones contextuales largas."
    )
    _text(fig, text, y=0.91, size=9.6)
    _classification_table(fig, report_dict, labels, [0.13, 0.44, 0.74, 0.18])
    _heatmap(fig, matrix, labels, "Matriz de confusion del baseline", [0.22, 0.08, 0.56, 0.30])
    pdf.savefig(fig)
    plt.close(fig)


def lora_section(pdf: PdfPages, evidence: dict, lora_matrix):
    fig = _page(pdf, "3. Solucion avanzada: DistilBERT + LoRA")
    lora = evidence["lora"]
    text = (
        f"Modelo base: {evidence['model_name']}. Se entrena un clasificador de cuatro clases "
        "sobre representaciones contextualizadas generadas por Transformer. En cada capa, los "
        "attention heads calculan relaciones entre tokens; LoRA congela los pesos preentrenados "
        "y aprende actualizaciones de bajo rango en las proyecciones q_lin y v_lin.\n\n"
        f"Configuracion: rank (r)={lora['r']}, alpha={lora['alpha']} (alpha/r=2), "
        f"dropout={lora['dropout']}, bias='none', learning_rate=2e-5, batch de train=16, "
        "batch de evaluacion=32, weight_decay=0.01 y 3 epocas. Se conserva el mejor checkpoint "
        "segun F1 macro de validacion.\n\n"
        f"El modelo tiene {evidence['total_parameters']:,} parametros totales y solo "
        f"{evidence['trainable_parameters']:,} entrenables ({evidence['trainable_percentage']:.2f}%). "
        f"El entrenamiento tomo {evidence['training_seconds'] / 60:.2f} minutos en una GPU GTX 1650 Ti. "
        "Durante cada paso, la perdida de clasificacion se propaga mediante backpropagation; "
        "el optimizador actualiza solo las matrices LoRA y mantiene congelados los embeddings y "
        "pesos originales. La validacion selecciono la epoca 2 (F1 macro=0.8776); la epoca 3 "
        "mantuvo practicamente el mismo rendimiento (0.8771), sin evidencia de una mejora sustancial."
    )
    _text(fig, text, y=0.91, size=9.8)
    _heatmap(fig, lora_matrix, LABELS, "Matriz de confusion LoRA en test", [0.22, 0.08, 0.56, 0.32])
    pdf.savefig(fig)
    plt.close(fig)


def comparison(pdf: PdfPages, baseline_matrix, lora_matrix, baseline_seconds: float,
               baseline_weighted, lora_weighted):
    fig = _page(pdf, "4. Evaluacion comparativa sobre el mismo test")
    rows = [
        ["TF-IDF + LogisticRegression", "0.8960", "0.8960", f"{baseline_weighted[2]:.4f}"],
        ["DistilBERT + LoRA", "0.8875", "0.8874", f"{lora_weighted[2]:.4f}"],
    ]
    ax = fig.add_axes([0.06, 0.73, 0.90, 0.14])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Modelo", "Accuracy", "F1 macro", "F1 ponderado"],
                     loc="center", cellLoc="center", colWidths=[0.42, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    text = (
        "Lectura. TF-IDF supera a LoRA por 0.0086 puntos de F1 macro (0.96% relativo). El test "
        "esta perfectamente balanceado (500 ejemplos por clase), por lo que F1 ponderado y F1 "
        "macro coinciden en ambos modelos; aun asi se reporta F1 ponderado explicitamente, como "
        "buena practica frente al desbalance de clases, en lugar de defender el proyecto solo con "
        "accuracy. Sports es la clase mas robusta en ambos modelos. LoRA mejora recall de Sci_Tech "
        "(0.88 frente a 0.87), pero pierde precision/recall en Business y no compensa el "
        "diferencial global.\n\n"
        f"Eficiencia. El baseline requirio {baseline_seconds:.2f} segundos para reajustar TF-IDF "
        "+ LogisticRegression sobre train (excluyendo el preprocesamiento NLP, ya calculado), "
        "mientras LoRA requirio 266.00 segundos de GPU para 3 epocas de fine-tuning. LoRA reduce "
        "la cantidad de parametros actualizados a 1.09%, pero no elimina el costo de ejecutar un "
        "Transformer ni su cadena de dependencias (GPU, tokenizador, PEFT). El baseline obtiene "
        "mejor F1 (0.8960 vs. 0.8874) y menor costo medido; por eso es la eleccion recomendada "
        "para este corpus."
    )
    _text(fig, text, y=0.65, size=9.4)
    _heatmap(fig, baseline_matrix, LABELS, "TF-IDF", [0.09, 0.09, 0.38, 0.30])
    _heatmap(fig, lora_matrix, LABELS, "LoRA", [0.53, 0.09, 0.38, 0.30])
    pdf.savefig(fig)
    plt.close(fig)


def interpretability(pdf: PdfPages, example):
    fig = _page(pdf, "5. Interpretabilidad: que ve cada modelo")
    text_snippet, true_label, predicted_label = example
    text = (
        "El baseline TF-IDF es una caja transparente: cada prediccion puede explicarse por los "
        "pesos que la regresion logistica asigna a los terminos y bigramas presentes en el texto. "
        "Su limite estructural es que ignora el orden de las palabras mas alla de los bigramas y no "
        "desambigua por contexto extendido.\n\n"
        "Ejemplo real de error del baseline en el test (misma observacion usada para evaluar ambos "
        f"modelos):\n\"{text_snippet}...\"\n"
        f"Etiqueta real: {true_label} | Prediccion del baseline: {predicted_label}\n\n"
        "Lectura. El texto mezcla vocabulario tipico de ambas categorias (empresas, productos y "
        "tecnologia), por lo que TF-IDF, al ponderar terminos de forma independiente, se inclina "
        "hacia la clase con mayor peso lexico agregado. Un Transformer con self-attention puede, "
        "en principio, atender simultaneamente al sujeto de la oracion, al verbo principal y al "
        "contexto discursivo (por ejemplo, si la nota describe una decision corporativa o un avance "
        "tecnico), resolviendo la ambiguedad mediante relaciones de largo alcance que el bag-of-words "
        "no representa.\n\n"
        "Honestidad de la evidencia. En esta corrida, DistilBERT + LoRA no elimino toda la confusion "
        "Business/Sci_Tech (ver matrices de la seccion 4): la mejora de recall en Sci_Tech sugiere que "
        "captura parte de esta señal, pero no se dispone de la prediccion puntual del modelo LoRA para "
        "este ejemplo especifico, por lo que no se afirma que lo haya resuelto; se documenta como el "
        "tipo de caso donde la atencion contextual deberia aportar valor."
    )
    _text(fig, text, size=9.6)
    pdf.savefig(fig)
    plt.close(fig)


def conclusions(pdf: PdfPages, baseline_seconds: float):
    fig = _page(pdf, "6. Conclusiones, limitaciones y referencias")
    text = (
        "Conclusiones. En AG News, un problema de cuatro clases con vocabulario muy distintivo, "
        "TF-IDF + LogisticRegression fue la mejor decision tecnica: alcanza F1 ponderado 0.8960 con "
        "una arquitectura compacta, interpretable, facil de serializar y con un ajuste medido en "
        f"{baseline_seconds:.2f} segundos. DistilBERT + LoRA demuestra adaptacion eficiente en "
        "parametros, porque solo actualiza 1.09% de ellos, pero su F1 ponderado de 0.8874 no supera "
        "al baseline ni en metricas ni en tiempo de ajuste del modelo. La semantica contextual no "
        "aporta una ventaja medible cuando los terminos y bigramas ya separan bien las categorias.\n\n"
        "Limitaciones. Se dispone de una sola corrida LoRA (sin repeticiones con distintas semillas) "
        "y no se registro consumo de memoria de GPU, por lo que la comparacion de eficiencia es "
        "parcial. El test es balanceado y relativamente pequeno; no permite estudiar robustez ante "
        "drift, clases raras o textos fuera de dominio. Ademas, no se conserva el modelo LoRA "
        "serializado, por lo que la seccion de interpretabilidad no pudo contrastar una prediccion "
        "puntual de LoRA sobre el mismo ejemplo del baseline. La matriz de confusion es evidencia de "
        "comportamiento agregado, no una explicacion causal de la atencion.\n\n"
        "Mejoras futuras. Repetir con varias semillas y registrar tiempo, memoria y latencia de ambos "
        "modelos; explorar r, alpha, learning rate y max_length; calibrar probabilidades; evaluar "
        "textos mas ambiguos y aplicar explicabilidad local (por ejemplo, perturbacion de tokens o "
        "Integrated Gradients). Para una eleccion productiva tambien deben medirse costo por prediccion "
        "y requisitos de despliegue.\n\n"
        "Referencias. Zhang et al. (2015), Character-level Convolutional Networks for Text Classification. "
        "Hu et al. (2022), LoRA: Low-Rank Adaptation of Large Language Models. Sanh et al. (2019), "
        "DistilBERT, a distilled version of BERT. Documentacion de scikit-learn, Hugging Face Transformers "
        "y PEFT. Dataset AG News, corpus de noticias de AG utilizado con fines academicos."
    )
    _text(fig, text, size=9.7)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    (labels, baseline_matrix, baseline_report_dict, baseline_seconds,
     baseline_weighted, example) = _baseline_confusion()
    lora_weighted = _parse_weighted_avg(evidence["lora_classification_report"])
    lora_matrix = pd.DataFrame([[411, 61, 6, 22], [40, 438, 1, 21],
                                 [6, 3, 488, 3], [36, 14, 12, 438]]).to_numpy()
    with PdfPages(OUTPUT_PATH) as pdf:
        cover(pdf, evidence)
        problem_and_pipeline(pdf)
        baseline(pdf, labels, baseline_matrix, baseline_report_dict)
        lora_section(pdf, evidence, lora_matrix)
        comparison(pdf, baseline_matrix, lora_matrix, baseline_seconds, baseline_weighted, lora_weighted)
        interpretability(pdf, example)
        conclusions(pdf, baseline_seconds)
    print(f"Informe generado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
