import tensorflow as tf 
from keras import layers,models
import numpy as np
import os
import cv2
import random
import shutil

width= 300
height = 300

ruta_train = 'IMAGENES/train/'

ruta_predict = 'IMAGENES/predict/'

# Configuración: si DELETE_BAD es True se eliminarán archivos corruptos,
# si es False se moverán a BAD_DIR (más seguro).
BAD_DIR = 'IMAGENES/bad_images'
DELETE_BAD = False

os.makedirs(BAD_DIR, exist_ok=True)

def handle_bad_file(path):
    try:
        if DELETE_BAD:
            os.remove(path)
            print(f"Deleted bad image: {path}")
        else:
            dest = os.path.join(BAD_DIR, os.path.basename(path))
            # Si ya existe, añade sufijo numérico
            if os.path.exists(dest):
                base, ext = os.path.splitext(os.path.basename(path))
                i = 1
                while os.path.exists(os.path.join(BAD_DIR, f"{base}_{i}{ext}")):
                    i += 1
                dest = os.path.join(BAD_DIR, f"{base}_{i}{ext}")
            shutil.move(path, dest)
            print(f"Moved bad image to: {dest}")
    except Exception as e:
        print(f"Warning: failed to remove/move bad file {path}: {e}")
'''
#ENTRENAMIENTO

train_x = []
train_y = []


for i in os.listdir(ruta_train):
    folder = os.path.join(ruta_train, i)
    if not os.path.isdir(folder):
        continue
    for j in os.listdir(folder):
        path = os.path.join(folder, j)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not read image: {path}")
            handle_bad_file(path)
            continue
        try:
            resized_image = cv2.resize(img, (width, height))
        except cv2.error as e:
            print(f"Warning: resize failed for {path}: {e}")
            handle_bad_file(path)
            continue

        train_x.append(resized_image)

        # Etiquetas simples 0/1
        if i.lower().startswith('cat'):
            train_y.append(0)
        else:
            train_y.append(1)

x_data = np.array(train_x)
y_data = np.array(train_y)

# Mostrar distribución de clases antes del entrenamiento
if y_data.size > 0:
    unique, counts = np.unique(y_data, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Class distribution (label:count): {dist}")

# Normalizar píxeles como en predicción y mezclar los datos
x_data = x_data.astype('float32') / 255.0
perm = np.random.permutation(len(x_data))
x_data = x_data[perm]
y_data = y_data[perm]

# Entrenamiento del modelo
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Iteraciones de entrenamiento
epochs = 20

model.fit(x_data, y_data, epochs = epochs)

models.save_model(model, 'modeloclasif.keras')

'''




#PREDICCION

# Cargar modelo y comprobar existencia de la imagen antes de redimensionar
model = models.load_model('modeloclasif.keras')

my_image_path = 'IMAGENES/test/12274.jpg'
my_image = cv2.imread(my_image_path)
if my_image is None:
    print(f"Error: no se pudo leer la imagen de predicción: {my_image_path}")
    handle_bad_file(my_image_path)
    raise SystemExit(1)
my_image = cv2.resize(my_image, (width, height))

# Normalizar igual que en entrenamiento (si corresponde) y obtener resultado legible
img_input = my_image.astype('float32') / 255.0
resultados = model.predict(np.array([img_input]))
prob = float(resultados.ravel()[0])
print("Raw prediction array:", resultados)
print(f"Probabilidad clase 1 (sigmoid): {prob:.4f}")
# Mapea la probabilidad a etiquetas legibles (ajusta si tu clase positiva es otra)
label = 'Dog' if prob >= 0.5 else 'Cat'
print("Etiqueta predicha:", label)



