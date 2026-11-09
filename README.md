# Projet Reconnaissance Faciale

Pipeline complet de reconnaissance faciale en temps réel : collecte de données, extraction d'embeddings avec **MobileFaceNet**, classification **k-NN / One-Class SVM**, et daemon de verrouillage d'écran (**FaceLock**) sur Linux.

## Pipeline

```
collecte_data.py ──► data/<personne>/*.jpg
                          │
                          ▼
              generer_embeddings.py  (MobileFaceNet TFLite, 192D)
                          │
                          ▼
              embeddings.pkl / embeddings.csv
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   creer_profil.py                 train_svm.py
   (profil k-NN + seuil)           (One-Class SVM)
          │                               │
          └───────────────┬───────────────┘
                          ▼
        inference_knn.py / inference.py  (webcam temps réel)
                          │
                          ▼
        facelock_daemon.py + facelock.service  (verrouillage auto)
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `collecte_data.py` | Capture webcam → `data/<classe>/` |
| `recuperation.py` | Récupération / augmentation de données |
| `generer_embeddings.py` | Embeddings MobileFaceNet (192D) |
| `creer_profil.py` | Profil k-NN + seuil de similarité cosinus |
| `train_svm.py` | One-Class SVM sur les embeddings |
| `train.py` | CNN binaire (approche alternative) + export TFLite |
| `inference.py` | Temps réel : MobileFaceNet + SVM |
| `inference_knn.py` | Temps réel : MobileFaceNet + k-NN (sans SVM) |
| `test_modele.py` | Test rapide du modèle |
| `facelock_daemon.py` | Daemon de verrouillage Linux |
| `facelock.service` | Unité systemd |

## Installation

```bash
git clone https://github.com/togbeabelmarcos-prog/projet_reconnaissance_fasciale.git
cd projet_reconnaissance_fasciale

python -m venv recon_fas
# Linux/macOS
source recon_fas/bin/activate
# Windows
recon_fas\Scripts\activate

pip install -r requirements.txt
```

### Modèle MobileFaceNet

Télécharger `MobileFaceNet.tflite` (~5 Mo) et le placer dans `model/` :

- [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)
- ou tout dépôt de poids MobileFaceNet TFLite 112x112

> Les poids et les embeddings ne sont **pas** versionnés (fichiers volumineux / données biométriques personnelles).

## Utilisation

```bash
# 1. Collecter des images (webcam)
python collecte_data.py

# 2. Générer les embeddings
python generer_embeddings.py

# 3. Créer le profil k-NN
python creer_profil.py

# 4. Reconnaissance temps réel
python inference_knn.py     # k-NN
python inference.py         # SVM
```

### FaceLock (Linux)

```bash
sudo cp facelock.service /etc/systemd/system/
sudo systemctl enable --now facelock
```

Adapter les chemins (`User`, `WorkingDirectory`, `ExecStart`) dans `facelock.service` à ta machine.

## Dépendances

- Python 3.10+
- TensorFlow (ou `tflite-runtime` pour l'inférence légère)
- OpenCV, NumPy, scikit-learn, MTCNN, tqdm

## Notes

- Les chemins dans les scripts (`MODEL_PATH`, etc.) sont absolus : à adapter à ton environnement.
- `data/`, les embeddings et les poids sont ignorés par git — à régénérer localement.
