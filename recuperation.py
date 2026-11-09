import cv2
import os
from mtcnn import MTCNN

input_folder = r'C:\Users\Marcos\Documents\data\data\Marcos'
output_folder = r'C:\Users\Marcos\Documents\data\data\recup_marcos'
    
detector = MTCNN()


for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
                
            # Chargement de l'image
            image_to_detect = cv2.imread(image_path)
            if image_to_detect is None:
                print(f"Erreur lors du chargement de l'image {image_path}")
                continue

            # Convertit l'image en RGB pour MTCNN
            rgb_image = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2RGB)

            # Détection des visages
            try:
                faces = detector.detect_faces(rgb_image)
                print(f"{len(faces)} visage(s) détecté(s) dans {file}")

                # Enregistrement des visages détectés
                for index, face in enumerate(faces):
                    x, y, width, height = face['box']
                    x, y = max(0, x), max(0, y)  # Assure que les coordonnées sont positives

                    # Découpe le visage détecté
                    current_face_image = image_to_detect[y:y+height, x:x+width]
                    face_image_path = os.path.join(output_subfolder, f'{os.path.splitext(file)[0]}_face_{index+1}.jpg')

                    # Sauvegarde l'image du visage détecté
                    cv2.imwrite(face_image_path, current_face_image)
                    print(f"Enregistrement du visage {index+1} dans {face_image_path}")
            except ValueError as e:
                print(f"Erreur lors du traitement de {file}: {str(e)}")
                continue