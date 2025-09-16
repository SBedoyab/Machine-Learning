# Descripción General
## [EDA](data_exploration/exploration.md)
## [Preprocesado](preprocessing/cleaning.md)
## [Creación de Características](feature_creation/features.md)
## [Modelado ML](model_building/model.md)
## [Conclusión](conclusion/conclusion.md)

---

## **Práctica Machine Learning: Predicción de Precio en Airbnb**

## Resumen
Este repositorio contiene el desarrollo de un proyecto de regresión, que tiene como objetivo predecir precios de anuncios de Airbnb's en una ciudad específica (Madrid). Se prepararon datos reales, se generaron variables categóricas/derivadas, se creó por buenas prácticas un modelo baseline tipo Decision Tree y un Random Forest como modelo Machine Learning final.

Las métricas principales fueron:
- **Decision Tree** (baseline): MAE ± 20.21 Euros en test.
- **Random Forest**: MAE ± 17.11 Euros en test.

> Para mayor comprensión del flujo de pensamiento, mirar el notebook `Práctica_ML_Santiago_Bedoya_Builes.ipynb`

## Contenido del repositorio
- **Data Exploration:** análisis exploratorio de los datos, gráficas descriptivas y primeras interpretaciones.
- **Preprocessing:** transformaciones y limpieza del dataset (eliminación de columnas, imputaciones, tratamiento de valores faltantes.
- **Feature Creation:** generación de variables derivadas y codificación de categóricas
- **Model Building:** construcción de modelos, esquema de validación y resultados obtenidos.
- **Conclusion:** principales hallazgos, limitaciones y apéndice de trazabilidad.

## Librerías utilizadas
- [Python Standard Library](https://docs.python.org/3/library/index.html): Módulos de Python.
- [Numpy](https://numpy.org/): Álgebra y manejo de arrays.  
- [Pandas](https://pandas.pydata.org/): Manipulación y análisis de datos.  
- [Matplotlib](https://matplotlib.org/): Visualización de gráficas estáticas.  
- [Seaborn](https://seaborn.pydata.org/): Visualización estadística especializada.  
- [Scikit-learn](https://scikit-learn.org/): Modelado y utilidades de machine learning.  

> Nota: El enunciado original es material privado de la academia KeepCoding® y no se ha reproducido de manera literal en este repositorio.