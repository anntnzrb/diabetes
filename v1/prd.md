# PRD: Implementación de Clasificación Supervisada para Predicción de Diabetes

## 1. Información General del Proyecto

**Entorno de Desarrollo:** Jupyter Notebook / Google Colab  
**Lenguaje:** Python  
**Algoritmo Principal:** Árboles de Decisión (Decision Tree)  
**Dataset:** diabetes.csv (768 registros × 9 columnas)

## 2. Objetivo Principal

Aplicar técnicas de clasificación supervisada utilizando el algoritmo de árboles de decisión (Decision Tree) sobre un conjunto de datos real de diabetes, desarrollando habilidades para el preprocesamiento de datos, entrenamiento de modelos, análisis de resultados y presentación de conclusiones mediante herramientas de programación en Python y visualización en Google Colab.

## 3. Instrucciones Específicas de la Consigna

### 3.1 Fase 1: Exploración y Comprensión del Problema
- **Leer cuidadosamente** el contexto de la práctica y **comprender el objetivo del estudio**
- **Revisar el conjunto de datos** proporcionado en la práctica (diabetes.csv)
- **Elegir las variables relevantes** para el análisis

### 3.2 Fase 2: Desarrollo en Google Colab
- **Cargar el dataset** y realizar **limpieza básica** si es necesario
- **Estandarizar o codificar variables** si el modelo lo requiere
- **Dividir el dataset** en conjuntos de entrenamiento y prueba
- **Entrenar el modelo de Árbol de Decisión** (DecisionTreeClassifier de scikit-learn)
- **Evaluar el modelo** usando **precisión, matriz de confusión y gráfica del árbol**
- **Realizar ajustes** si el modelo presenta bajo rendimiento
- **Probar con otro modelo** y si es más adecuado, **especificarlo**

## 4. Contexto del Dataset de Diabetes

### 4.1 Características del Dataset
- **Archivo:** diabetes.csv
- **Dimensiones:** 768 registros × 9 columnas
- **Objetivo:** Predecir si un paciente tiene diabetes (clasificación binaria)

### 4.2 Variables del Dataset
**Variables Predictoras:**
- **pregnancies:** Número de veces que la persona ha estado embarazada
- **glucose:** Concentración de glucosa en plasma a 2 horas en una prueba oral de tolerancia a la glucosa
- **diastolic:** Presión arterial diastólica (mm Hg)
- **triceps:** Espesor del pliegue de la piel del tríceps (mm)
- **insulin:** Suero de insulina de 2 horas (mu U/ml)
- **bmi:** Índice de masa corporal (peso en kg/(altura en m)²)
- **dpf:** Pedigree de la función de la diabetes (probabilidad de diabetes basada en historia familiar)
- **age:** Edad (años)

**Variable Objetivo:**
- **diabetes:** Variable binaria (0 = no tiene diabetes, 1 = tiene diabetes)

## 5. Desarrollo Detallado

### 5.1 Fase 1: Exploración y Comprensión del Problema

#### 5.1.1 Carga y Análisis Inicial
- Cargar el dataset usando pandas
- Examinar estructura general del dataset
- Verificar existencia de valores nulos
- Analizar distribución de la variable objetivo (balance de clases)

#### 5.1.2 Análisis Exploratorio de Datos
- Generar estadísticas descriptivas
- Crear visualizaciones de distribución (histogramas, boxplots)
- **Crear matriz de correlación** entre variables
- **Generar mapa de calor** para visualizar correlaciones
- Identificar outliers mediante gráficos de caja

#### 5.1.3 Análisis de Variables Específicas
- Examinar rango de valores de cada variable
- Identificar variables con **valores 0 que podrían ser faltantes** (glucose, diastolic, triceps, insulin, bmi no pueden ser realmente 0)
- Evaluar importancia de cada variable para la predicción

### 5.2 Fase 2: Desarrollo en Google Colab

#### 5.2.1 Preprocesamiento Específico para Dataset de Diabetes

**Manejo de Valores Faltantes:**
- **Reemplazar valores 0 por NaN** en columnas donde 0 no es válido: glucose, diastolic, triceps, insulin, bmi
- **Imputar valores NaN con la media** de cada columna
- Mantener valores 0 válidos en pregnancies (una mujer puede no haber estado embarazada)

**Limpieza y Transformación:**
- Verificar tipos de datos
- Aplicar estandarización si es necesario para el modelo
- **Decisión sobre manejo de outliers** (documentar si se eliminan o mantienen)

#### 5.2.2 División del Dataset
- Separar variables predictoras (X) de variable objetivo (y)
- Usar **train_test_split** con proporción 70-30 (entrenamiento-prueba)
- Establecer **random_state** para reproducibilidad

#### 5.2.3 Entrenamiento del Modelo Principal

**Implementación Obligatoria:**
- Usar **DecisionTreeClassifier de scikit-learn**
- Entrenar con datos de entrenamiento
- Configurar parámetros básicos del árbol

#### 5.2.4 Evaluación Obligatoria del Modelo

**Métricas Requeridas:**
- **Calcular precisión (accuracy)** en entrenamiento y prueba
- **Generar matriz de confusión**
- **Crear gráfica del árbol de decisión**
- Mostrar importancia de variables

**Visualizaciones Obligatorias:**
- Gráfico del árbol de decisión usando plot_tree
- Matriz de confusión visual
- Gráfico de importancia de características

#### 5.2.5 Optimización y Modelos Alternativos

**Si el modelo presenta bajo rendimiento:**
- Ajustar hiperparámetros del DecisionTreeClassifier (max_depth, min_samples_split, etc.)
- **Probar modelos alternativos:**
  - Random Forest
  - Regresión Logística  
  - Naive Bayes
- **Especificar cuál modelo es más adecuado** y justificar la elección
- Comparar métricas entre modelos

**Librerías y Dependencias:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
# Modelos alternativos si son necesarios:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
```

## 6. Estructura Obligatoria del Notebook

### 6.1 Secciones Requeridas
1. **Importación de librerías**
2. **Carga del dataset diabetes.csv**
3. **Exploración inicial y comprensión del problema**
4. **Análisis exploratorio de datos con visualizaciones**
5. **Preprocesamiento específico (manejo de 0s como NaN, imputación con media)**
6. **División en entrenamiento y prueba**
7. **Entrenamiento de DecisionTreeClassifier**
8. **Evaluación con precisión, matriz de confusión y gráfica del árbol**
9. **Ajustes del modelo si presenta bajo rendimiento**
10. **Prueba de modelos alternativos si es necesario**
11. **Conclusiones y especificación del modelo más adecuado**

### 6.2 Documentación Requerida
- Comentarios explicativos en cada celda de código
- Markdown cells explicando decisiones de preprocesamiento
- Interpretación de resultados y visualizaciones
- Justificación de la elección del modelo final

## 7. Criterios de Éxito Específicos

### 7.1 Cumplimiento de Instrucciones
- ✅ Uso obligatorio de DecisionTreeClassifier de scikit-learn
- ✅ Evaluación con precisión, matriz de confusión y gráfica del árbol
- ✅ Si rendimiento es bajo, probar otros modelos y especificar el más adecuado
- ✅ Preprocesamiento adecuado del dataset de diabetes

### 7.2 Calidad Técnica
- Precisión mínima aceptable ≥ 70%
- Código limpio, comentado y reproducible
- Visualizaciones claras y bien etiquetadas
- Manejo adecuado de valores faltantes específicos del dataset

### 7.3 Análisis y Decisiones
- Interpretación correcta de la estructura del árbol
- Justificación de decisiones de preprocesamiento
- Análisis de importancia de variables
- Especificación clara del modelo más adecuado con justificación

## 8. Consideraciones Técnicas

### 8.1 Especificidades del Dataset de Diabetes
- Reconocer que valores 0 en glucose, diastolic, triceps, insulin, bmi son probablemente faltantes
- Aplicar imputación con media como estrategia de manejo de faltantes
- Mantener 0s válidos en pregnancies

### 8.2 Reproducibilidad
- Usar random_state fijo en train_test_split y modelos
- Asegurar que el notebook sea ejecutable en Google Colab

### 8.3 Decisión de Modelo
- Si DecisionTreeClassifier no alcanza rendimiento aceptable, implementar alternativas
- Documentar claramente cuál modelo es superior y por qué

Este PRD ahora incluye las instrucciones específicas de la consigna y incorpora el conocimiento específico del dataset de diabetes de la guía, asegurando que un agente pueda seguir exactamente lo que se requiere hacer.
