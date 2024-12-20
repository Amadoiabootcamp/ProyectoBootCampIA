# ProyectoBootCampIA
Proyecto final
# Análisis de Sentimiento de Reseñas de Productos

Este proyecto implementa un modelo de aprendizaje automático para analizar el sentimiento de reseñas de productos en español. Se utiliza regresión logística para clasificar las reseñas como positivas o negativas, basándose en su contenido textual.

## Características
- **Limpieza del texto**: Se utiliza una lista personalizada de stopwords en español.
- **Vectorización**: Conversión de texto en vectores numéricos utilizando TF-IDF.
- **Modelo de clasificación**: Regresión logística para predecir el sentimiento.
- **Evaluación**: Métricas como precisión y reporte de clasificación.

## Requisitos

Para ejecutar el proyecto, asegúrate de tener las siguientes bibliotecas instaladas:

- pandas
- scikit-learn

Puedes instalarlas utilizando pip:
```bash
pip install pandas scikit-learn
```

## Estructura de los Datos

El conjunto de datos de entrada es un diccionario que contiene:
- **`review`**: Texto de la reseña del producto.
- **`sentiment`**: Etiqueta del sentimiento (1 para positiva, 0 para negativa).

Ejemplo:
```python
{
    'review': [
        'Este producto es increíble, lo recomiendo mucho',
        'Muy mala calidad, no lo volvería a comprar'
    ],
    'sentiment': [1, 0]
}
```

## Flujo del Proyecto
1. **Preparación de Datos**:
   - Conversión del diccionario a un DataFrame de pandas.
   - División del conjunto de datos en entrenamiento y prueba.

2. **Preprocesamiento del Texto**:
   - Vectorización usando TF-IDF, con soporte para stopwords personalizadas.

3. **Entrenamiento del Modelo**:
   - Se entrena un modelo de regresión logística con los datos vectorizados.

4. **Evaluación**:
   - Se calcula la precisión del modelo y se genera un reporte detallado de clasificación.

5. **Predicción del Usuario**:
   - El usuario ingresa una nueva reseña y el modelo predice su sentimiento.

## Uso

1. Clona este repositorio.
2. Ejecuta el script principal. Por ejemplo:
```bash
python sentiment_analysis.py
```
3. Ingresa una reseña cuando se te solicite.

El programa clasificará la reseña como **Positiva** o **Negativa**.

## Ejemplo

Entrada del usuario:
```
Ingresa tu reseña de producto para analizar el sentimiento: Me encantó el producto, es fantástico
```

Salida:
```
La reseña es clasificada como: Positiva
```

## Métricas de Evaluación

El modelo utiliza las siguientes métricas para evaluar el rendimiento:
- **Precisión**: Proporción de predicciones correctas.
- **Reporte de Clasificación**: Incluye precisión, recall y F1-score para cada clase.

## Notas

- La lista de stopwords incluye palabras comunes en español, así como términos adicionales relevantes al dominio (por ejemplo, "pesimo", "malo", etc.).
- Puedes ampliar la lista de stopwords o ajustar los parámetros del modelo para mejorar el rendimiento.

## Licencia

Este proyecto está disponible bajo la licencia MIT.
