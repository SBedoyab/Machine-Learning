## [Descripción General](../README.md)
## [EDA](../data_exploration/exploration.md)
# Preprocesado
## [Creación de Características](../feature_creation/features.md)
## [Modelado ML](../model_building/model.md)
## [Conclusión](../conclusion/conclusion.md)

---

## Limpieza y Preprocesado

Esta sección describe transformaciones realizadas al dataset antes de la [creación de variables](../feature_creation/features.md) y el [modelado](../model_building/model.md). Se documenta el paso a paso de las decisiones y referencias a las celdas del notebook `Práctica_ML_Santiago_Bedoya_Builes.ipynb`.

## 1. Eliminación de columnas con baja cobertura o irrelevantes

### Eliminación de `Square Feet`
- Bajos registros, ± solo 4% de datos representativos
- Alta variabilidad.

Resultado: reducción de ruido y riesgo de imputaciones sesgadas.

> Notebook: Python cells #7-#9.

## 2. Tratamiento de outliers y normalizaciones

### Acotamiento de Outliers en `Minimum Nights`
- Existencia de valores extremos >31 noches poco representativos.
- Acción: Se sustituye por la mediana = 2.
   ```python
   train['Minimum Nights'] = np.where(train['Minimum Nights'] > 31,
                                       train['Minimum Nights'].median(),
                                       train['Minimum Nights'])
   ```

Resultado: Normalización exitosa de `Minimum Nights` que evitan sesgo.

> Notebook: Python cell #12.

## 3. Consolidación de categorías

### Agrupación de tipos de propiedad `Property Type`.
- Muchas categorías con pocos registros.
- Acción: fusión en la categoría “Other”.
   ```python
   train['Property Type'] = np.where(train['Property Type'].isin(['Apartment', 'House', 'Condominium', 'Bed & Breakfast', 'Loft', 'Other']),
                                    train['Property Type'],
                                    'Other')
   ```
Resultado: Reducción considerable de futura alta cardinalidad

> Notebook: Python cell #14.

### Simplificación de tipos de cama `Bed Type`
- Motivo: mayoría de registros con `Real Bed`.
- Acción: se crea variable binaria: `Real Bed` vs `Other Bed Type`.
   ```python
   train['Bed Type'] = np.where(train['Bed Type'] == 'Real Bed',
                                train['Bed Type'],
                                'Other Bed Type')
   ```
Resultado: Reducción considerable de futura alta cardinalidad

> Notebook: Python cell #16.

## 4. Codificación de variables categóricas

### One-hot encoding
- Prepararación de variables categóricas para el modelado.
- Acción: aplicación de `pd.get_dummies` sobre las variables aptas al encoding.

## 5. Filtrado de variables redundantes

### Colinealidad
- Se detectaron correlaciones >0,80 en matriz de correlación.
- Acción: eliminación de `Room Type_Private room` y `Beds` por redundancia.

## 6. Imputaciones y transformaciones adicionales

### Valores nulos
- La mayoría de variables presentan >95% de cobertura.
- `Security Deposit` ±43% no nulos y `Cleaning Fee` ±59% no nulos se mantienen para etapas posteriores de ingeniería de variables.
   - Se hizo una imputación importante, donde en estas dos variables anteriores se llenaron con 0 bajo el argumento que al ser variables tan importantes, no es viable que 'por olvido' al anfitrion del Airbnb se le olvide digitar.

### Comodidades `Amenities`
- Se genera arbitrariamente una lista de comodidades consideradas clave, para convertir estas a variables binarias. `['Air conditioning', 'Essentials', 'Free parking on premises', 'Pool', 'Wireless Internet', 'Heating', 'Gym', 'Hot tub', 'Kitchen']`
- Columna `Amenities` original se elimina.

## 7. Observaciones finales del preprocesado

- Dataset reducido a variables más informativas, controlando colinealidad y categorías raras.
- Columnas totales finales del dataset: 29/89 → 67% cardinalidad reducida.
- Dataset preparado para la siguiente etapa de [Creación de Características](../feature_creation/features.md).