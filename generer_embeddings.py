"""
Génère les embeddings MobileFaceNet pour toutes les images de /data_recup/moi/
Sauvegarde les embeddings dans un fichier CSV + pickle
"""
import os
import csv
import pickle
import numpy as np
import cv2
from tqdm import tqdm

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────
DATA_DIR        = r"C:\Users\Marcos\Documents\data\data\recup_marcos"
MODEL_PATH      = r"C:\Users\Marcos\Documents\data\model\MobileFaceNet.tflite"
OUTPUT_CSV = r"C:\Users\Marcos\Documents\data\model\embeddings.csv"
OUTPUT_PICKLE = r"C:\Users\Marcos\Documents\data\model\embeddings.pkl"

# ──────────────────────────────────────────
# CHARGEMENT MODÈLE
# ──────────────────────────────────────────
print(" Chargement MobileFaceNet...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(" Modèle chargé\n")

# ──────────────────────────────────────────
# FONCTION D'EXTRACTION
# ──────────────────────────────────────────
def get_embedding(image_path):
    """
    Lit une image de data_recup/moi/ (crop rectangulaire MTCNN ~150x190)
    Fait center crop carré + resize 112x112
    Retourne l'embedding 192D normalisé
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Les images de data_recup/moi/ sont des crops rectangulaires de MTCNN
    h, w = img.shape[:2]
    
    # Center crop carré
    size = min(h, w)
    y1, y2 = (h - size) // 2, (h - size) // 2 + size
    x1, x2 = (w - size) // 2, (w - size) // 2 + size
    
    img_crop = img[y1:y2, x1:x2]
    img_crop = cv2.resize(img_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
    
    # BGR → RGB
    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    
    # Normaliser [-1, 1]
    img_norm = (img_rgb.astype(np.float32) - 127.5) / 127.5
    
    # Batch de 2 images identiques
    batch = np.stack([img_norm, img_norm], axis=0)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], batch)
    interpreter.invoke()
    sorties = interpreter.get_tensor(output_details[0]['index'])
    
    # Normaliser L2
    vecteur = sorties[0].copy()
    norme = np.linalg.norm(vecteur)
    if norme > 0:
        vecteur = vecteur / norme
    
    return vecteur


# ──────────────────────────────────────────
# PARCOURS DES IMAGES
# ──────────────────────────────────────────
print(f" Dossier : {DATA_DIR}")
images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
print(f"   {len(images)} images trouvées\n")

if not images:
    print(" Aucune image trouvée !")
    exit(1)

# ──────────────────────────────────────────
# GÉNÉRATION DES EMBEDDINGS
# ──────────────────────────────────────────
print(" Génération des embeddings...")
embeddings = []
images_valides = []

for img_name in tqdm(images, desc="Processing"):
    img_path = os.path.join(DATA_DIR, img_name)
    emb = get_embedding(img_path)
    
    if emb is not None:
        embeddings.append(emb)
        images_valides.append(img_name)
    else:
        print(f"     Image ignorée : {img_name}")

print(f"\n {len(embeddings)} embeddings générés avec succès")

# ──────────────────────────────────────────
# STATISTIQUES
# ──────────────────────────────────────────
print("\n STATISTIQUES")
print(f"   Images totales    : {len(images)}")
print(f"   Images valides    : {len(embeddings)}")
print(f"   Taux de réussite  : {len(embeddings)/len(images)*100:.1f}%")

# Cohérence des embeddings
if len(embeddings) > 1:
    n = len(embeddings)
    similarities = []
    for i in range(min(n, 500)):
        for j in range(i+1, min(n, 500)):
            sim = np.dot(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    if similarities:
        print(f"\n   Similarité moyenne (échantillon) : {np.mean(similarities):.4f}")
        print(f"   Similarité min    : {np.min(similarities):.4f}")
        print(f"   Similarité max    : {np.max(similarities):.4f}")

# ──────────────────────────────────────────
# SAUVEGARDE
# ──────────────────────────────────────────
print(f"\n Sauvegarde CSV → {OUTPUT_CSV}")
header = ['image_name'] + [f'emb_{i}' for i in range(192)]

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for img_name, emb in zip(images_valides, embeddings):
        row = [img_name] + emb.tolist()
        writer.writerow(row)

print(f" CSV sauvegardé ({len(embeddings)} lignes, {len(header)} colonnes)")

# Pickle pour compatibilité
print(f"\n Sauvegarde Pickle → {OUTPUT_PICKLE}")
data = {
    'embeddings': np.array(embeddings),
    'images': images_valides,
    'n_embeddings': len(embeddings),
    'embedding_dim': len(embeddings[0]) if embeddings else 0
}

with open(OUTPUT_PICKLE, 'wb') as f:
    pickle.dump(data, f)

print(f" Pickle sauvegardé")
print(f"\n Prochaine étape : python train_svm.py")
