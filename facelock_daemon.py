#!/usr/bin/env python3
"""
facelock_daemon.py - Daemon de surveillance faciale 24h/24

Fonctionnalités :
- Tourne en continu depuis le boot
- Capture des frames par intervalles
- Verrouille quand l'utilisateur n'est plus là
- Déverrouille quand l'utilisateur revient

Usage :
    sudo systemctl start facelock
    sudo systemctl enable facelock  # Au démarrage
"""
import os
import sys
import time
import gc  # Garbage collector
import pickle
import numpy as np
import cv2
from mtcnn import MTCNN
import subprocess
import logging
from datetime import datetime

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
LOG_FILE    = "/tmp/facelock_daemon.log"  # /tmp est toujours accessible

# Seuil de reconnaissance
SEUIL_KNN = 0.75

# Timing (secondes)
CHECK_INTERVAL = 2          # Vérifier toutes les 2 secondes
LOCK_DELAY = 5              # Verrouiller après 5s sans visage
UNLOCK_DELAY = 2            # Déverrouiller après 2s avec visage

# Utilisateur à surveiller (celui qui a lancé la session)
TARGET_USER = os.environ.get("USER", "nicaise")

# ──────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────
def load_resources():
    """Charge le modèle et le profil"""
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()
        outp = interpreter.get_output_details()
        
        with open(PROFIL_PATH, 'rb') as f:
            profil = pickle.load(f)
        
        embeddings_dataset = profil['tous_embeddings']
        k = profil.get('k', 100)
        detector = MTCNN()
        
        logger.info("Ressources chargées avec succès")
        return interpreter, inp, outp, embeddings_dataset, k, detector
    except Exception as e:
        logger.error(f"Erreur chargement: {e}")
        return None

# ──────────────────────────────────────────
# EXTRACTION VISAGE
# ──────────────────────────────────────────
def extraire_visage(image_bgr, detector):
    """Extrait le visage avec MTCNN"""
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
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_bgr.shape[1], x + w), min(image_bgr.shape[0], y + h)
    
    visage = image_rgb[y1:y2, x1:x2]
    
    if visage.size == 0:
        return None
    
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
def get_embedding(visage_rgb_112, interpreter, inp, outp):
    """Calcule l'embedding 192D"""
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
def reconnaitre_knn(embedding, embeddings_dataset, k=100):
    """Compare avec les k plus proches du dataset"""
    similarites = [float(np.dot(embedding, emb)) for emb in embeddings_dataset]
    similarites.sort(reverse=True)
    top_k = similarites[:k]
    return np.mean(top_k)

# ──────────────────────────────────────────
# LOCK/UNLOCK
# ──────────────────────────────────────────
def is_session_locked():
    """
    Vérifie si la session est verrouillée
    Retourne True si l'écran de login GDM est affiché
    """
    try:
        # Méthode 1: Vérifier si gnome-screensaver est actif
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.screensaver", "active"],
            capture_output=True, text=True, timeout=2
        )
        if "true" in result.stdout.lower():
            return True
        
        # Méthode 2: Vérifier si l'écran de veille est activé
        result2 = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.session", "idle-delay"],
            capture_output=True, text=True, timeout=2
        )
        
        # Méthode 3: Vérifier avec loginctl si session est active
        result3 = subprocess.run(
            ["loginctl", "show-session", "-p", "LockedHint"],
            capture_output=True, text=True, timeout=2
        )
        if "yes" in result3.stdout.lower():
            return True
            
        # Par défaut, on assume que si on a appelé lock_session(), c'est verrouillé
        # On utilise un fichier flag pour tracker l'état
        flag_file = "/tmp/facelock_locked"
        if os.path.exists(flag_file):
            return True
            
        return False
    except Exception as e:
        logger.debug(f"Erreur is_session_locked: {e}")
        return False


# Variable globale pour tracker l'état de verrouillage
_session_locked_flag = False


def lock_session():
    """Verrouille la session"""
    global _session_locked_flag
    logger.info(" VERROUILLAGE de la session")
    try:
        # Méthode 1: xdg-screensaver (le plus universel)
        result = subprocess.run(
            ["xdg-screensaver", "lock"],
            timeout=5, capture_output=True
        )
        if result.returncode == 0:
            _session_locked_flag = True
            with open("/tmp/facelock_locked", "w") as f:
                f.write("locked")
            logger.info("Session verrouillée (xdg-screensaver)")
            return
    except Exception as e:
        logger.debug(f"xdg-screensaver échoué: {e}")
    
    try:
        # Méthode 2: dbus direct
        subprocess.run([
            "dbus-send", "--type=method_call",
            "--dest=org.gnome.ScreenSaver",
            "/org/gnome/ScreenSaver",
            "org.gnome.ScreenSaver.Lock"
        ], timeout=5, capture_output=True)
        _session_locked_flag = True
        with open("/tmp/facelock_locked", "w") as f:
            f.write("locked")
        logger.info("Session verrouillée (dbus)")
    except Exception as e:
        logger.error(f"Échec verrouillage: {e}")


def unlock_session():
    """
    Déverrouille la session (réveille l'écran)
    NOTE: Ne fonctionne que si la session est déjà authentifiée
    """
    global _session_locked_flag
    logger.info(" DÉVERROUILLAGE de la session")
    try:
        # Méthode 1: xdotool pour simuler une touche (réveille l'écran)
        result = subprocess.run(["xdotool", "key", "space"], timeout=2,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Signal de déverrouillage envoyé (xdotool) - ret: {result.returncode}")
        
        # Réessaie avec Enter au cas où
        subprocess.run(["xdotool", "key", "Return"], timeout=2,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Méthode 2: dbus pour désactiver le screensaver
        subprocess.run([
            "dbus-send", "--type=method_call",
            "--dest=org.gnome.ScreenSaver",
            "/org/gnome/ScreenSaver",
            "org.gnome.ScreenSaver.SetActive",
            "boolean:false"
        ], timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Signal de déverrouillage envoyé (dbus)")
        
        # Nettoie le flag
        _session_locked_flag = False
        if os.path.exists("/tmp/facelock_locked"):
            os.remove("/tmp/facelock_locked")
            
    except Exception as e:
        logger.warning(f"Déverrouillage limité: {e}")
        # Nettoie quand même le flag
        _session_locked_flag = False
        if os.path.exists("/tmp/facelock_locked"):
            os.remove("/tmp/facelock_locked")

# ──────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("FaceLock Daemon - Démarrage")
    logger.info(f"Utilisateur cible: {TARGET_USER}")
    logger.info(f"Seuil k-NN: {SEUIL_KNN}")
    logger.info(f"Intervalle: {CHECK_INTERVAL}s")
    logger.info("=" * 60)
    
    # Charger les ressources
    resources = load_resources()
    if resources is None:
        logger.error("Impossible de charger les ressources, arrêt")
        sys.exit(1)
    
    interpreter, inp, outp, embeddings_dataset, k, detector = resources
    
    # Ouvrir la webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        logger.error("Webcam inaccessible")
        sys.exit(1)
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # État du daemon
    etat = "RECHERCHE"  # RECHERCHE | VERROUILLER | DEVERROUILLER
    temps_sans_visage = 0
    temps_avec_visage = 0
    dernier_lock = 0
    dernier_unlock = 0
    
    logger.info("Daemon prêt - En attente de visage...")
    
    frame_count = 0  # Compteur pour le nettoyage mémoire

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(CHECK_INTERVAL)
                continue
            
            frame_count += 1
            
            # Détection
            visage = extraire_visage(frame, detector)
            reconnu = False
            knn_score = 0
            
            if visage is not None:
                emb = get_embedding(visage, interpreter, inp, outp)
                knn_score = reconnaitre_knn(emb, embeddings_dataset, k)
                reconnu = knn_score >= SEUIL_KNN
                
                # LOG TOUJOURS le score pour debug
                logger.info(f" Visage détecté | k-NN: {knn_score:.3f} | Seuil: {SEUIL_KNN} | Reconnu: {reconnu}")
            
            # Vérifie l'état de la session
            session_verrouillee = is_session_locked()
            
            # Machine à états
            if reconnu:
                temps_sans_visage = 0
                temps_avec_visage += CHECK_INTERVAL
                
                if session_verrouillee and temps_avec_visage >= UNLOCK_DELAY:
                    # Déverrouiller SEULEMENT si verrouillé
                    unlock_session()
                    etat = "DEVERROUILLER"
                    dernier_unlock = time.time()
                    temps_avec_visage = 0
                    logger.info(f"✓ Utilisateur reconnu → DÉVERROUILLAGE (k-NN: {knn_score:.3f})")
                elif not session_verrouillee:
                    # Session déjà ouverte - rien à faire
                    etat = "DEVERROUILLER"
                    logger.debug(f"✓ Utilisateur reconnu | Session déjà ouverte")
                
                logger.debug(f"Reconnu | État: {etat} | Temps: {temps_avec_visage}s")
            else:
                temps_avec_visage = 0
                temps_sans_visage += CHECK_INTERVAL
                
                if not session_verrouillee and temps_sans_visage >= LOCK_DELAY:
                    # Verrouiller SEULEMENT si pas déjà verrouillé
                    lock_session()
                    etat = "RECHERCHE"
                    dernier_lock = time.time()
                    temps_sans_visage = 0
                    logger.warning(f"✗ Utilisateur absent → VERROUILLAGE")
                
                if visage is None:
                    logger.debug(f"Aucun visage | État: {etat}")
                else:
                    logger.debug(f"Pas reconnu (k-NN: {knn_score:.3f}) | État: {etat}")
            
            # Nettoyage mémoire toutes les 30 itérations
            if frame_count % 30 == 0:
                del frame, visage, emb, knn_score
                gc.collect()
                
                # Log mémoire occasionnel
                import resource
                mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                logger.debug(f" Mémoire: {mem_mb:.0f}MB")

            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        logger.info("Daemon arrêté par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        logger.info("Daemon terminé")

if __name__ == "__main__":
    main()
