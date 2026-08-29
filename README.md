# Pre-entrega 3: Clasificador supervisado con TF-IDF

Pipeline de clasificación de noticias de AG News mediante preprocesamiento NLP, representación TF-IDF y regresión logística.

## 📁 Estructura del Repositorio

```text
.
├── data/
│   ├── ag_news_train.csv
│   └── ag_news_test.csv
├── notebooks/
│   └── pipeline_clasificador_tfidf.ipynb
├── modelo_tfidf_ag_news.joblib
├── requirements.txt
└── README.md
```

## Ejecución

Instalar las dependencias:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Abrir y ejecutar todas las celdas de [`notebooks/pipeline_clasificador_tfidf.ipynb`](notebooks/pipeline_clasificador_tfidf.ipynb), desde la carga de datos hasta la serialización del modelo. El notebook detecta la raíz del proyecto y utiliza los CSV ubicados en `data/`.

El archivo [`modelo_tfidf_ag_news.joblib`](modelo_tfidf_ag_news.joblib) contiene el pipeline entrenado y permite realizar nuevas predicciones sin volver a ajustar el vectorizador.

## Preprocesamiento

Se incorporan las operaciones desarrolladas en el Módulo 2 para AG News, cuyo idioma es inglés:

- eliminación de HTML, URLs y caracteres no alfabéticos;
- normalización de espacios y conversión a minúsculas;
- tokenización y lematización con `en_core_web_sm`;
- eliminación de puntuación, espacios y tokens de un solo carácter.

Estas transformaciones se aplican por igual a train y test, pero no aprenden parámetros a partir del test.

### Pipeline y prevención de data leakage

1. Se cargan los splits provistos de AG News (`8.000` registros de train y `2.000` de test).
2. El train se divide internamente en ajuste y validación estratificada (`80/20`).
3. Se comparan dos configuraciones de `TfidfVectorizer` usando únicamente ese ajuste y validación:
	- `max_features=10000`, `ngram_range=(1, 1)`.
	- `max_features=20000`, `ngram_range=(1, 2)`.
4. La mejor configuración se vuelve a ajustar sobre todo el train. El test se procesa solo con `transform`, nunca con `fit`.

Se usa `stop_words="english"` para quitar palabras funcionales frecuentes que suelen aportar poco a la clasificación. Esta decisión reduce el vocabulario y el costo de la matriz, pero se mantiene la información específica del dominio, como `AP`/`ap` cuando identifica a Associated Press.

### Modelo y configuración

Se eligió `LogisticRegression` como baseline porque funciona bien con matrices dispersas TF-IDF, es rápido de entrenar y permite establecer una referencia interpretable antes de probar modelos más complejos. También se configuró `min_df=2`, `max_df=0.95`, `sublinear_tf=True` y `max_iter=1000`.

Naive Bayes también es una alternativa rápida para texto, pero supone independencia entre las palabras y los n-gramas. SVM suele ser competitiva, aunque sus puntuaciones son menos directas de interpretar y no proporciona probabilidades sin calibración adicional. Por estos motivos, la regresión logística ofrece un buen equilibrio entre rendimiento, simplicidad e interpretabilidad.

En la ejecución de referencia, la configuración seleccionada fue `max_features=20000` con unigramas y bigramas (`ngram_range=(1, 2)`), con F1 macro de validación de `0.8927`. Las matrices finales tienen forma `X_train=(8000, 20000)` y `X_test=(2000, 20000)`, y el resultado en test fue accuracy `0.8960` y F1 macro `0.8960`.

### Análisis preliminar

Según la matriz de confusión y el `classification_report`, `Business` es la categoría más difícil de predecir, con F1 `0.85`, seguida de `Sci_Tech`, con F1 `0.87`. La principal confusión se produce entre ambas: 39 noticias reales de `Business` fueron clasificadas como `Sci_Tech` y 46 noticias reales de `Sci_Tech` fueron clasificadas como `Business`. Esto se explica por el vocabulario compartido sobre empresas, mercados, tecnología y resultados financieros.

`World` alcanza F1 `0.90` y presenta algunas confusiones con `Business`, mientras que `Sports` es la categoría más fácil de identificar, con F1 `0.96` y 482 predicciones correctas de 500.

### Archivo de requerimientos

Las dependencias y versiones mínimas se encuentran en [`requirements.txt`](requirements.txt). Para este módulo se utilizan principalmente `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `spacy` y `joblib`. `transformers` y `nltk` corresponden a los módulos de tokenización anteriores.

### Carga del pipeline guardado

```python
import joblib

pipeline = joblib.load("modelo_tfidf_ag_news.joblib")
prediccion = pipeline.predict(["A new article about science and technology"])
```