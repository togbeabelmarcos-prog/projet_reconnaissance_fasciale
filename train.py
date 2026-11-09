import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime

# Fonction pour charger les images depuis deux dossiers (moi et autres)
def load_images_from_folders(data_dir, target_size=(128, 128)):
    """
    Charge les images depuis data_dir/moi et data_dir/autres
    Retourne X, y où y=1 pour moi, y=0 pour autres
    """
    images = []
    labels = []
    
    # Dossier moi
    moi_dir = os.path.join(data_dir, 'moi')
    if os.path.exists(moi_dir):
        for f in os.listdir(moi_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(moi_dir, f)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, target_size)
                    images.append(img_resized)
                    labels.append(1)  # classe moi = 1
    else:
        print(f" Dossier {moi_dir} introuvable")

    # Dossier autres
    autres_dir = os.path.join(data_dir, 'autres')
    if os.path.exists(autres_dir):
        for f in os.listdir(autres_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(autres_dir, f)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, target_size)
                    images.append(img_resized)
                    labels.append(0)  # classe autres = 0
    else:
        print(f" Dossier {autres_dir} introuvable")
    
    return np.array(images), np.array(labels)

# Fonction pour normaliser les images
def preprocess_images(images):
    return images / 255.0

# Fonction pour obtenir le répertoire des logs pour TensorBoard
def get_tensorboard_log_dir():
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join(os.getcwd(), "logs", f"fit_{today_date}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

if __name__ == "__main__":

    data_dir = os.path.join(os.getcwd(), "data_recup")
    
    # Charger les images (on peut splitter train/val plus tard ou avoir des dossiers séparés)
    # Pour simplifier, on va charger toutes les images et faire un split manuel
    X, y = load_images_from_folders(data_dir, target_size=(128, 128))
    
    if len(X) == 0:
        print(" Aucune image chargée. Vérifie les dossiers.")
        exit()
    
    print(f" Images chargées : {len(X)}")
    print(f"   - moi : {np.sum(y==1)}")
    print(f"   - autres : {np.sum(y==0)}")
    
    # Normaliser
    X = preprocess_images(X)
    
    # Split train/validation (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n Train : {X_train.shape[0]} images")
    print(f" Validation : {X_val.shape[0]} images")
    
    # Définir le modèle CNN (on peut réduire la complexité car binaire)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # sortie binaire
    ])
    
    # Compiler
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # Callbacks
    log_dir = get_tensorboard_log_dir()
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Early stopping pour éviter surapprentissage
    early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    # Entraînement
    history = model.fit(X_train, y_train,
                        epochs=50,
                        validation_data=(X_val, y_val),
                        callbacks=[tensorboard_callback, early_stop],
                        batch_size=32)
    
    # Sauvegarder le modèle
    model.save('modele_binaire_128.h5')
    print(" Modèle sauvegardé : modele_binaire_128.h5")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('modele_binaire_128.tflite', 'wb') as f:
        f.write(tflite_model)
    print(" Modèle TFLite sauvegardé : modele_binaire_128.tflite")
    
    print(f" Logs TensorBoard dans {log_dir}")