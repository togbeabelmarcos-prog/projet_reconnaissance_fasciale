"""
Inference en temps réel avec MobileFaceNet + SVM
Utilise MTCNN pour l'extraction de visage
"""
import cv2
import numpy as np
import pickle
import os
import time
from mtcnn import MTCNN

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────
MODEL_PATH  = r"C:\Users\Marcos\Documents\data\model\MobileFaceNet.tflite"
PROFIL_PATH = os.path.expanduser(r"C:\Users\Marcos\Documents\data\model\facelock\profil_complet.pkl")

# ──────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────
print(" Chargement modèle + profil...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()

with open(PROFIL_PATH, "rb") as f:
    profil = pickle.load(f)

seuil = profil['seuil_cosinus']
print(f" Prêt — seuil cosinus : {seuil:.4f}\n")

# MTCNN
detector = MTCNN()

# ──────────────────────────────────────────
# EXTRACTION VISAGE
# ──────────────────────────────────────────
def extraire_visage(image_bgr):
    """
    Extrait le visage avec MTCNN - MÊME MÉTHODE QUE recuperation.py
    Puis fait center crop carré + resize 112x112 pour MobileFaceNet
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        faces = detector.detect_faces(image_rgb)
    except Exception as e:
        # MTCNN plante parfois sur des images problématiques
        return None
    
    if not faces:
        return None
    
    face = max(faces, key=lambda f: f['confidence'])
    x, y, w, h = face['box']
    
    # Vérifier que le visage est assez grand
    if w < 50 or h < 50:
        return None
    
    # MÊME CROP QUE recuperation.py (sans marge supplémentaire !)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_bgr.shape[1], x + w), min(image_bgr.shape[0], y + h)
    
    visage = image_rgb[y1:y2, x1:x2]
    
    # Vérifier que le crop n'est pas vide
    if visage.size == 0:
        return None
    
    # Maintenant: center crop carré PUIS resize à 112x112
    # (car les images de recuperation.py sont rectangulaires ~150x190)
    vh, vw = visage.shape[:2]
    size = min(vh, vw)
    vy1, vy2 = (vh - size) // 2, (vh - size) // 2 + size
    vx1, vx2 = (vw - size) // 2, (vw - size) // 2 + size
    
    visage_crop = visage[vy1:vy2, vx1:vx2]
    
    # Vérifier que le crop carré n'est pas vide
    if visage_crop.size == 0:
        return None
    
    visage_crop = cv2.resize(visage_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
    
    return visage_crop

# ──────────────────────────────────────────
# EMBEDDING
# ──────────────────────────────────────────
def get_embedding(visage_rgb_112):
    """Calcule l'embedding 192D normalisé"""
    v = (visage_rgb_112.astype(np.float32) - 127.5) / 127.5
    batch = np.stack([v, v], axis=0)
    interpreter.set_tensor(inp[0]['index'], batch)
    interpreter.invoke()
    vec = interpreter.get_tensor(out[0]['index'])[0].copy()
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

# ──────────────────────────────────────────
# RECONNAISSANCE
# ──────────────────────────────────────────
def reconnaitre(embedding):
    """Test avec cosinus + SVM"""
    # C1 : similarité cosinus
    sim = float(np.dot(profil['vecteur_reference'], embedding))
    c1_ok = sim >= seuil
    
    # C2 : SVM
    e_sc = profil['svm_scaler'].transform([embedding])
    c2_ok = profil['svm_model'].predict(e_sc)[0] == 1
    svm_dst = float(profil['svm_model'].decision_function(e_sc)[0])
    
    return c1_ok and c2_ok, sim, svm_dst, c1_ok, c2_ok

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
def main():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cam.isOpened():
        print(" Webcam inaccessible")
        return
    
    print("╔══════════════════════════════════════════╗")
    print("║    INFÉRENCE TEMPS RÉEL — MTCNN + SVM    ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n  Seuil cosinus : {seuil:.4f}")
    print(f"  Appuie sur 'q' pour quitter\n")
    
    frame_count = 0
    reconnu_count = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Extraction
        visage = extraire_visage(frame)
        
        if visage is None:
            cv2.putText(frame, "Aucun visage", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Embedding + reconnaissance
            emb = get_embedding(visage)
            reconnu, sim, svm_dst, c1, c2 = reconnaitre(emb)
            
            if reconnu:
                reconnu_count += 1
                label = f"MOI {sim:.2f}"
                color = (0, 255, 0)
            else:
                label = f"AUTRE {sim:.2f}"
                color = (0, 0, 255)
            
            cv2.putText(frame, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Stats
            if frame_count % 30 == 0:
                taux = reconnu_count / frame_count * 100
                print(f"  Frames: {frame_count} | Reconnu: {taux:.1f}%")
        
        cv2.imshow("FaceLock", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
    print(f"\n Session terminée")
    print(f"   Frames totaux : {frame_count}")
    print(f"   Frames reconnus : {reconnu_count}")
    print(f"   Taux reconnaissance : {reconnu_count/frame_count*100:.1f}%")

if __name__ == "__main__":
    main()
