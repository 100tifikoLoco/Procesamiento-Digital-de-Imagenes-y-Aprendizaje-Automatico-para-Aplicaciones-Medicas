#PCA prupuesta por Díaz Sánchez Luis Manuel

#Librerias
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D #Para graficar en 3D
import numpy as np
import os
import cv2
from sklearn.decomposition import PCA
from tqdm import tqdm # Barra de progreso
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Numero de imágenes procesadas para actualizar la barra
update_frequency = 200

#Directorio de las imagenes a utilizar
image_dir = "data/image/clean"

#Creacion de lista para guardado de imágenes y etiquetas
image_data = []
labels = []

#Preprocesamiento - Normalización y aumento de contraste
def preprocess_image(image):
    #Normalización (escalar entre 0 y 1)
    image = image / 255.0
    #Aumento de contraste (ecualización de histograma adaptativa)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply((image * 255).astype(np.uint8)) #Convertir a uint8 para clahe
    return image

#Iteracion para cargar preprocesar imagenes
for i, file_name in enumerate(tqdm(os.listdir(image_dir), desc="Procesando Imágenes")):
    if i % update_frequency == 0:
        tqdm.write(f"Procesadas {i} imágenes") # Mostrar el progreso
    file_path = os.path.join(image_dir, file_name)
    #Verificar si el archivo es una imagen
    if os.path.isfile(file_path):
        #Cargar imagenes y redimensionar  
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (64,64)) #Ajuste de tamaño
            image = preprocess_image(image)
            image_data.append(image.flatten()) #Covertir imagen a vector
            labels.append(file_name) #Usa el nombre de la carpeta como etiqueta
            
X = np.array(image_data)
y = np.array(labels)

#Normalización extra para PCA
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Datos cargados correctamente - {X.shape} imágenes")

#Tamaño de la muestra de puntos(imágenes) a mostrar en la grafica
sample_size = 500
sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
print(f"Indices seleccionados - {sample_indices[:10]}")
X_s = X[sample_indices] #Subconjunto reducido

#Aplicación de PCA
n_components = 75 #Definición de número de componentes principales
pca =  PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

#Clustering con k-means
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca[sample_indices, :50])
custom_colors = ['green' if cluster==0 else 'red' for cluster in clusters]
plt.figure(figsize=(8,6))
plt.scatter(X_pca[sample_indices,0], X_pca[sample_indices, 1], c=custom_colors, cmap='coolwarm', edgecolors='k', s=50)
plt.title('Clustering sobre imágenes de Ultrasonido Pulmonar')
plt.show()
