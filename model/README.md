# model/

Fichiers non versionnés (poids + données biométriques).

À placer ici :

| Fichier | Origine |
|---|---|
| `MobileFaceNet.tflite` (~5 Mo) | Modèle pré-entraîné 112×112 → embedding 192D |
| `embeddings.pkl` / `embeddings.csv` | Générés par `generer_embeddings.py` |
| `profil_complet.pkl` | Généré par `creer_profil.py` |

MobileFaceNet : [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)
