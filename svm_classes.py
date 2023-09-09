# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:35:18 2023

@author: luimungar2
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


class MalwareClassifier:
    def __init__(self, data_path, feature_path):
        self.data_path = data_path
        self.feature_path = feature_path
        self.data = None
        self.caracteristicas = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.label_encoder = None
        self.clf = None
        
    def map_labels(self,label):
        if label == 'S':
            return 0
        elif label == 'B':
            return 1
        else:
            raise ValueError("Etiqueta desconocida: {}".format(label))

    def load_data(self):
        # Cargar los datos desde un archivo CSV
        self.data = pd.read_csv(self.data_path)
        self.caracteristicas = pd.read_csv(self.feature_path)
        print("Características: ", self.caracteristicas)
        self.clases = ["B", "S"]
        print("Clases: ", self.clases)

    def preprocess_data(self):
        # Reemplazar los valores '?' con NaN
        self.data.replace('?', float('nan'), inplace=True)
        # Eliminar filas con valores NaN
        self.data.dropna(inplace=True)
        self.X = self.data.drop('class', axis=1)
        self.y = self.data['class']

    def split_data(self):
        # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=42)

    def train_svm(self):
        # Crear el clasificador SVM
        self.svm = SVC(kernel='linear')
        # Entrenar el modelo SVM
        self.svm.fit(self.X_train, self.y_train)

    def predict_svm(self):
        # Predecir las etiquetas en el conjunto de prueba
        self.y_pred = self.svm.predict(self.X_test)

    def evaluate_svm(self):
        # Estadísticas de Kappa
        kappa = cohen_kappa_score(self.y_test, self.y_pred)
        # Error absoluto medio
        y_test_num = np.array(list(map(self.map_labels, self.y_test)))
        y_pred_num = np.array(list(map(self.map_labels, self.y_pred)))
        mae = mean_absolute_error(y_test_num, y_pred_num)
        # Error cuadrático medio
        rmse = np.sqrt(mean_squared_error(y_test_num, y_pred_num))

        # Imprimir los resultados
        num_features = self.X_test.shape[1]
        num_instances = self.X_test.shape[0]
        num_malware = self.y_test[self.y_test == 'S'].shape[0]
        num_goodware = self.y_test[self.y_test == 'B'].shape[0]
        num_correct = np.sum(self.y_test == self.y_pred)
        num_incorrect = np.sum(self.y_test != self.y_pred)

        print("Número de atributos (Características):", num_features)
        print("Número de muestras:", num_instances)
        print("Número de malware en el dataset:", num_malware)
        print("Número de goodware en el dataset:", num_goodware)
        print("Muestras correctamente clasificadas:", num_correct)
        print("Muestras erróneamente clasificadas:", num_incorrect)
        
        print("Coeficiente Kappa:", kappa)
        print("Error Medio Absoluto:", mae)
        print("Error Cuadrático Medio:", rmse)
        # Calcular la precisión del modelo
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Precisión del modelo SVM: {:.2f}%".format(accuracy * 100))
        # Graficar la clasificación de las muestras
        x = [-2, 2, -2, 2]  # Coordenadas x
        y = [-2, -2, 2, 2]  # Coordenadas y
        labels = ['Goodware bien clasificado', 'Goodware clasificado como malware', 'Malware clasificado como goodware', 'Malware bien clasificado']  # Etiquetas de las clasificaciones

        # Crear una figura y un conjunto de ejes
        fig, ax = plt.subplots()

        # Configurar los límites de los ejes
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

        # Trazar los puntos en los cuadrantes correspondientes
        for i in range(len(x)):
            ax.scatter(x[i], y[i], label=labels[i], alpha=1)

        # Agregar dispersión alrededor de los puntos centrales
        for i in range(len(x)):
            # Calcular la cantidad de puntos a agregar para bien clasificados y fallos de clasificación
            num_points = num_instances
            if 'bien clasificado' in labels[i]:
                num_points = num_correct
            elif 'clasificado como' in labels[i]:
                num_points = num_incorrect
                
            # Generar puntos aleatorios con distribución normal
            points_x = np.random.normal(x[i], 0.1, num_points)
            points_y = np.random.normal(y[i], 0.1, num_points)
            ax.scatter(points_x, points_y, marker='x', alpha=0.025, color='grey')

        # Agregar etiquetas a los ejes
        ax.set_xlabel('Goodware <---> Malware')
        ax.set_ylabel('Goodware <---> Malware')

        # Agregar título al gráfico
        ax.set_title('Clasificación de muestras')

        # Quitar los números de los ejes
        ax.set_xticks([])
        ax.set_yticks([])

        # Mostrar la leyenda
        ax.legend()

        # Mostrar el gráfico
        plt.show()

        # Convertir las etiquetas de clase a valores numéricos
        label_encoder = LabelEncoder()
        self.y_train_encoded = label_encoder.fit_transform(self.y_train)
        self.y_test_encoded = label_encoder.transform(self.y_test)

        # Crear un clasificador SVC con probability=True
        self.clf = SVC(probability=True)

        # Entrenar el clasificador con tus datos de entrenamiento
        self.clf.fit(self.X_train, self.y_train_encoded)

        # Calcular la probabilidad de predicción para la clase positiva (B) en lugar de la clase negativa (S)
        self.y_pred_prob = self.clf.predict_proba(self.X_test)[:, 1]

        # Calcular la curva ROC
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test_encoded, self.y_pred_prob)

        # Calcular el área bajo la curva ROC (AUC)
        self.roc_auc = auc(self.fpr, self.tpr)

        # Graficar la curva ROC
        plt.figure(figsize=(10, 6))
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.4f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.show()



# Crear una instancia de la clase MalwareClassifier
data_path = 'C:/Users/usuario/.spyder-py3/ciber/TFM/malware/drebin-215-dataset-5560malware-9476-benign.csv'  # Reemplaza con la ruta correcta a tu archivo de datos
feature_path = 'C:/Users/usuario/.spyder-py3/ciber/TFM/malware/dataset-features-categories.csv'  # Reemplaza con la ruta correcta a tu archivo de características
# Crear una instancia del clasificador de malware
classifier = MalwareClassifier(data_path, feature_path)

# Cargar los datos
classifier.load_data()

# Preprocesar los datos
classifier.preprocess_data()

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
classifier.split_data()

# Entrenar el clasificador SVM
classifier.train_svm()

# Predecir las etiquetas en el conjunto de prueba
classifier.predict_svm()

# Evaluar el rendimiento del clasificador SVM
classifier.evaluate_svm()




