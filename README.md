# proy_data_science_III
Proyecto Datascience modulo 3


# Pre-entrega: Pipeline de Entrenamiento, Validación y Clasificador Base

Este repositorio contiene la infraestructura inicial y el pipeline de entrenamiento en PyTorch diseñado para el proyecto integrador de Deep Learning.

## 📁 Estructura del Repositorio

```text
.
├── data/                  # Directorio para datasets locales
│   └── README.md
├── notebooks/             # Lógica del modelo y pipeline en Jupyter Notebook
│   └── pipeline_entrenamiento.ipynb
├── requirements.txt       # Archivo de dependencias del entorno
└── README.md              # Documentación del checkpoint


Documentación del Checkpoint
1. Versión de PyTorch y Configuración del Dispositivo
Versión de PyTorch: 2.0+ (verificada dinámicamente en tiempo de ejecución).

Detección de Dispositivo: El pipeline detecta automáticamente el hardware disponible mediante:

cuda si hay una GPU NVIDIA disponible.

mps para GPUs de Apple Silicon.

cpu como fallback predeterminado.

2. Arquitectura Base del Modelo
Se implementó un Multi-Layer Perceptron (MLP) modular mediante la clase BaseMLPClassifier (nn.Module):

Capa de Entrada: 20 características de entrada (nn.Linear(20, 64)).

Activación No Lineal: ReLU().

Regularización: Dropout(p=0.2) para prevenir overfitting.

Capa de Salida: 2 clases de salida (nn.Linear(64, 2)).

3. Hiperparámetros Seleccionados
Optimizador: Adam (torch.optim.Adam)

Learning Rate (lr): 0.001

Función de Pérdida: nn.CrossEntropyLoss()

Tamaño de Batch: 32

Épocas: 40 (con Early Stopping, patience=10, restauración de mejores pesos)

4. Interpretación de la Curva de Pérdida y Métricas
Descenso de Pérdida: La pérdida de entrenamiento disminuyó de forma consistente desde ~0.59 en la primera época hasta ~0.09 al finalizar las 40 épocas.

Comportamiento en Validación: La curva de pérdida en el conjunto de validación acompañó la tendencia del conjunto de entrenamiento de forma suave y sin divergencias, lo que confirma que el modelo está aprendiendo patrones generalizables y que zero_grad() y .backward() funcionan correctamente.

Early Stopping: Se implementó una clase EarlyStopping (patience=10) que monitorea val_loss y restaura automáticamente los pesos del mejor modelo si no hay mejora durante 10 épocas consecutivas. En esta corrida no llegó a activarse porque val_loss siguió mejorando durante las 40 épocas configuradas; actúa como salvaguarda ante overfitting en corridas futuras.

Accuracy: Se logró un Accuracy final >97% en el conjunto de validación.