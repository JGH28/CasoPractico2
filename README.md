# Universidad de Guadalajara
Centro Universitario de Ciencias Exactas e Ingenierías
Ing. Biomédica
Materia: Métodos biomédicos de IA
Profesor: Daniel Farfán

# Caso Práctico 2
## Predicción de Tipos de Reacción Química usando Fingerprints Moleculares:
Este proyecto se inspira en el trabajo de Schwaller et al. (2021), donde se utilizó una red neuronal basada en atención para mapear el espacio de reacciones químicas utilizando fingerprints moleculares.
https://github.com/rxn4chemistry/rxnfp 

Este proyecto se centra en el uso de fingerprints moleculares para predecir el tipo de reacción química a partir de las estructuras moleculares. Utilizamos un modelo de regresión logística que clasifica las reacciones químicas basándose en las características moleculares de los compuestos.


## Objetivos del Proyecto
* Crear un modelo de regresión logística para predecir el tipo de reacción química usando fingerprints moleculares.
* Evaluar el rendimiento del modelo mediante métricas como precisión, AUC-ROC y F1 Score.
* Implementar la técnica de Y-scrambling para validar la solidez del modelo.
* Visualizar fingerprints moleculares utilizando TMAP.


## Estructura del Proyecto
1. Preprocesamiento de Datos: Carga y limpieza del dataset, generación de fingerprints moleculares.
2. Entrenamiento del Modelo: Creación y ajuste del modelo de regresión logística.
3. Evaluación del Modelo: Evaluación de las predicciones del modelo y validación con Y-scrambling.
4. Visualización de Fingerprints con TMAP: Exploración topológica de las reacciones.

## Instalación
Para configurar el entorno de trabajo, es necesario instalar varias librerías específicas, como:
 `rdkit`, `tmap`.
 
### Requisitos
1. Python 3.6+
2. Librerías necesarias:
    rdkit (2020.03.3)
    tmap
    pandas
    scikit-learn

## Instalación usando Conda
Recomendamos el uso de conda para instalar las dependencias necesarias:

```bash
conda create -n reaccion-quimica python=3.8 -y
conda activate reaccion-quimica
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c tmap tmap -y
pip install pandas scikit-learn matplotlib
```

## ¿Cómo usarlo?

### Preprocesamiento de Datos
Carga y genera los fingerprints moleculares a partir de las reacciones del dataset:

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# Carga del dataset
df = pd.read_csv('rxnfp/data/schneider50k.tsv', sep='\t')

# Generación de fingerprints
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

example_rxn = "Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"

fp = rxnfp_generator.convert(example_rxn)
print(len(fp))
print(fp[:5])
```

### Entrenamiento del Modelo
Entrena el modelo de regresión logística con las características generadas:

```python 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['fingerprint'], df['rxn_class'], test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
```
### Visualización con TMAP
TMAP permite visualizar las similitudes entre reacciones químicas basadas en fingerprints moleculares:
```python
from tmap import LSHForest, Mapper

# Inicialización de TMAP
lsh_forest = LSHForest()
mapper = Mapper()

# Insertar fingerprints en TMAP
for fp in df['fingerprint']:
    mapper.add_fp(fp)

# Visualizar resultados
mapper.plot()
```

## Referencias
El dataset utilizado en este proyecto se basa en el trabajo de Schneider et al., que contiene más de 50,000 reacciones químicas. Los datos han sido procesados para generar fingerprints moleculares utilizando la librería rdkit.
Los fingerprints permiten explorar la estructura química de las reacciones y agruparlas en un espacio vectorial, ayudando en el proceso de clasificación.

### Conjunto Schneider 50k - tutorial

En los notebooks, se muestracómo generar un atlas interactivo de reacciones para el conjunto Schneider 50k. El resultado final es similar a este Atlas Interactivo de Reacciones. **[interactive Reaction Atlas](https://rxn4chemistry.github.io/rxnfp//tmaps/tmap_ft_10k.html)**.

En él encontrarás diferentes propiedades de reacciones destacadas en las distintas capas:

<div style="text-align: center">
<img src="nbs/images/tmap_properties.jpg" width="800" style="max-width: 800px">
<p style="text-align: center;"> <b>Figure:</b> Reaction atlas of 50k data set with different properties highlighted. </p>
</div>

## USPTO 1k TPL (conjunto de datos de clasificación de reacciones)

Se presenta un nuevo conjunto de datos para la clasificación de reacciones químicas llamado USPTO 1k TPL. USPTO 1k TPL se deriva de la [base de datos USPTO](https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873) por Lowe. Consiste en 445k reacciones divididas en 1000 etiquetas de plantillas. El conjunto de datos se dividió aleatoriamente en un 90% para entrenamiento/validación y un 10% para pruebas. Las etiquetas se obtuvieron mapeando átomos en el conjunto de datos USPTO con [RXNMapper](http://rxnmapper.ai), luego aplicando el [flujo de trabajo de extracción de plantillas](https://github.com/reymond-group/CASP-and-dataset-performance) por Thakkar et al., y finalmente seleccionando reacciones pertenecientes a los 1000 hashes de plantillas más frecuentes. Estos hashes de plantillas se tomaron como etiquetas de clase. De manera similar al conjunto de datos Pistachio, USPTO 1k TPL está fuertemente desbalanceado.

El conjunto de datos se puede descargar desde: [MappingChemicalReactions](https://ibm.box.com/v/MappingChemicalReactions).

