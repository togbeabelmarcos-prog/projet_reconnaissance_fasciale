import cv2
import time
import os

def collecter_images(classe, nombre_images=5000):
    """
    classe: "Marcos"
    """
    dossier = f"data/{classe}"  
    os.makedirs(dossier, exist_ok=True)
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erreur caméra")
        return
    
    print(f"\n COLLECTE POUR: {classe.upper()}")
    print(f"Dossier: {dossier}")
    print(f"Nombre d'images: {nombre_images}")
    
    if classe == "Marcos":
        print("\nTOI: Bouge naturellement, varie les poses")
    else:
        print("\n AUTRES: Change de personne régulièrement")
    
    input("\nAppuie sur Entrée pour commencer...")
    
    images_existantes = len([f for f in os.listdir(dossier) if f.startswith(classe) and f.endswith('.jpg')])
    img_counter = images_existantes
    print(f"\n {images_existantes} images déjà présentes -> on continue à partir de {img_counter}")
    
    duree_pause = 0.1  # 5 images/seconde
    
    while img_counter < nombre_images:
        ret, frame = cam.read()
        if not ret:
            break
        
        cv2.putText(frame, f"Classe: {classe}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {img_counter}/{nombre_images}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Collecte", frame)
        
        # Sauvegarde - CORRECTION: utilise toujours "data"

        img_name = os.path.join(dossier, f"{classe}_{img_counter:04d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f" {img_name}")
        
        img_counter += 1 
        time.sleep(duree_pause)
        
        if cv2.waitKey(1) == 27:  # Échap
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"\n Collecte terminée: {img_counter} images dans {dossier}")

# COLLECTE
# ======================
# COLLECTE MONO-CLASSE
# ======================

print("=" * 50)
print("COLLECTE  pour Marcos")
print("=" * 50)
print("📁 Les images seront sauvegardées dans data/Marcos/")

collecter_images("Marcos", nombre_images=3000,)

print("\nCollecte terminée ! Vérifie avec: tree data/")