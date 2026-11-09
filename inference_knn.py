"""
Inference temps réel avec MobileFaceNet + k-NN (SANS SVM)
Compare l'embedding webcam avec TOUS les embeddings du dataset
"""
import cv2
import numpy as np
import pickle
import os
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
MODEL_PATH  = "/home/nicaise/Bureau/cyber_agent/projet licence/model/MobileFaceNet.tflite"
PROFIL_PATH = os.path.expanduser("~/.facelock/profil_complet.pkl")

# ──────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────
print("⏳ Chargement modèle + profil...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
outp = interpreter.get_output_details()

with open(PROFIL_PATH, 'rb') as f:
    profil = pickle.load(f)

# Charger TOUS les embeddings du dataset
embeddings_dataset = profil['tous_embeddings']
n_dataset = len(embeddings_dataset)
k = profil.get('k', 100)

# SEUILS MANUELS - À AJUSTER SELON TES TESTS
# Ton k-NN max webcam était ~0.88, donc on met 0.75 pour être sûr
seuil_knn = 0.75  # Baisse si pas reconnu, monte si faux positifs
seuil_cosinus = 0.55

print(f" Prêt")
print(f"   Dataset : {n_dataset} embeddings")
print(f"   k-NN    : k={k}, seuil={seuil_knn:.4f}")
print(f"   Cosinus : seuil={seuil_cosinus:.4f}\n")

# MTCNN
detector = MTCNN()

# ──────────────────────────────────────────
# EXTRACTION VISAGE
# ──────────────────────────────────────────
def extraire_visage(image_bgr):
    """Extrait le visage avec MTCNN - MÊME MÉTHODE QUE recuperation.py"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        faces = detector.detect_faces(image_rgb)
    except Exception:
        return None
    
    if not faces:
        return None
    
    face = max(faces, key=lambda f: f['confidence'])
    x, y, w, h = face['box']
    
    if w < 50 or h < 50:
        return None
    
    # Crop MTCNN brut (comme recuperation.py)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_bgr.shape[1], x + w), min(image_bgr.shape[0], y + h)
    
    visage = image_rgb[y1:y2, x1:x2]
    
    if visage.size == 0:
        return None
    
    # Center crop carré + resize 112x112
    vh, vw = visage.shape[:2]
    size = min(vh, vw)
    vy1, vy2 = (vh - size) // 2, (vh - size) // 2 + size
    vx1, vx2 = (vw - size) // 2, (vw - size) // 2 + size
    
    visage_crop = visage[vy1:vy2, vx1:vx2]
    
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
    vec = interpreter.get_tensor(outp[0]['index'])[0].copy()
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# ──────────────────────────────────────────
# RECONNAISSANCE K-NN
# ──────────────────────────────────────────
def reconnaitre_knn(embedding, k=100):
    """
    Compare l'embedding avec TOUS ceux du dataset
    Retourne la similarité moyenne avec les k plus proches
    """
    # Similarités avec tous les embeddings du dataset
    similarites = [float(np.dot(embedding, emb)) for emb in embeddings_dataset]
    
    # Trier et prendre les k plus proches
    similarites.sort(reverse=True)
    top_k = similarites[:k]
    
    # Similarité moyenne des k plus proches
    knn_moy = np.mean(top_k)
    
    # Similarité maximale (meilleur match)
    best_match = similarites[0]
    
    return knn_moy, best_match


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
    
    # Test de lecture caméra
    ret, test_frame = cam.read()
    if not ret:
        print(" Impossible de lire la webcam")
        return
    print(f" Webcam OK - Frame: {test_frame.shape}")
    
    print("╔══════════════════════════════════════════╗")
    print("║    INFÉRENCE TEMPS RÉEL — k-NN (100)     ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n  Seuil k-NN    : {seuil_knn:.4f}")
    print(f"  Seuil cosinus : {seuil_cosinus:.4f}")
    print(f"  Appuie sur 'q' pour quitter\n")
    
    # Créer la fenêtre
    cv2.namedWindow("FaceLock k-NN", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FaceLock k-NN", 640, 480)
    
    frame_count = 0
    reconnu_count = 0
    knn_scores = []
    detection_count = 0

    print(" En attente de visage...\n")
    
    while True:
        ret, frame = cam.read()
        # Juste pour le diagnostic - à retirer ensuite
        if frame_count == 5:
            cv2.imwrite("/tmp/test_frame.jpg", frame)
            print(f"Frame sauvegardée dans /tmp/test_frame.jpg")
            print(f"Brightness moyenne : {frame.mean():.1f}")
            print(f"Shape : {frame.shape}")
        if not ret:
            continue
        
        frame_count += 1
        
        # Détection MTCNN
        visage = extraire_visage(frame)
        
        if visage is None:
            # Afficher "En attente" sur la frame
            cv2.putText(frame, "En attente de visage...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: En attente... (MTCNN ne détecte rien)")
        else:
            # Visage détecté !
            emb = get_embedding(visage)
            
            # Reconnaissance k-NN
            knn_moy, best = reconnaitre_knn(emb, k)
            knn_scores.append(knn_moy)
            
            # Décision
            reconnu = knn_moy >= seuil_knn
            
            if reconnu:
                reconnu_count += 1
                label = f"MOI {knn_moy:.2f}"
                color = (0, 255, 0)
            else:
                label = f"AUTRE {knn_moy:.2f}"
                color = (0, 0, 255)
            
            cv2.putText(frame, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Stats
            if frame_count % 30 == 0:
                taux = reconnu_count / frame_count * 100
                if knn_scores:
                    knn_mean = np.mean(knn_scores[-30:])
                    print(f"Frame {frame_count:4d} | Reconnu: {taux:5.1f}% | k-NN: {knn_mean:.3f} (seuil: {seuil_knn:.2f})")
        
        # Afficher la frame
        cv2.imshow("FaceLock k-NN", frame)
        
        # Attendre pour l'affichage Wayland
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
    # Bilan
    print("\n" + "="*60)
    print(" SESSION TERMINÉE")
    print("="*60)
    print(f"   Frames totaux     : {frame_count}")
    print(f"   Frames reconnus   : {reconnu_count}")
    print(f"   Taux reconnaissance : {reconnu_count/frame_count*100:.1f}%")
    print(f"   k-NN moyen        : {np.mean(knn_scores):.4f}")
    print(f"   k-NN min          : {np.min(knn_scores):.4f}")
    print(f"   k-NN max          : {np.max(knn_scores):.4f}")
    print("="*60)
    
    if np.mean(knn_scores) >= seuil_knn:
        print("\n   BONNE RECONNAISSANCE !")
    else:
        print("\n    Reconnaissance insuffisante")
        print(f"      k-NN moyen: {np.mean(knn_scores):.4f} < {seuil_knn:.4f}")

if __name__ == "__main__":
    main()
