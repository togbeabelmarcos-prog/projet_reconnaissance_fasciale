# Guide pas à pas — du début à la fin

Ce document décrit **chaque étape dans l'ordre** pour faire fonctionner le projet, depuis une machine vierge jusqu'à la reconnaissance faciale en temps réel (et le verrouillage automatique sur Linux).

---

## Étape 0 — Prérequis

- **Python 3.10 ou +** : vérifier avec `python --version`
- **Git** : vérifier avec `git --version`
- Une **webcam** connectée
- (Linux uniquement) accès `sudo` pour le daemon FaceLock

---

## Étape 1 — Récupérer le projet

```bash
cd C:\Users\Marcos\Documents      # ou ton dossier de travail
git clone https://github.com/togbeabelmarcos-prog/projet_reconnaissance_fasciale.git
cd projet_reconnaissance_fasciale
```

---

## Étape 2 — Créer l'environnement virtuel et installer les dépendances

```bash
python -m venv recon_fas
```

Activer le venv :

```bash
# Windows
recon_fas\Scripts\activate
# Linux / macOS
source recon_fas/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Vérification : `pip list` doit afficher `tensorflow`, `opencv-python`, `scikit-learn`, `mtcnn`, `tqdm`.

---

## Étape 3 — Télécharger le modèle MobileFaceNet

1. Télécharger `MobileFaceNet.tflite` (~5 Mo) depuis [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF) (ou tout miroir de poids TFLite 112×112).
2. Le placer dans le dossier `model/` du projet :
   ```
   projet_reconnaissance_fasciale/
   └── model/
       └── MobileFaceNet.tflite
   ```

> ⚠️ Sans ce fichier, **aucune** étape suivante ne fonctionnera.

---

## Étape 4 — Adapter les chemins dans les scripts

Les scripts contiennent des **chemins absolus** (parfois Windows, parfois Linux). Ouvre chaque fichier et corrige les constantes en haut du fichier avec **tes** chemins réels :

| Fichier | Constante(s) à modifier | Valeur attendue |
|---|---|---|
| `collecte_data.py` | `dossier` (ligne ~9) | `data/<TonNom>/` |
| `recuperation.py` | `input_folder`, `output_folder` | `data/Marcos/` → `data/recup_marcos/` |
| `generer_embeddings.py` | `DATA_DIR`, `MODEL_PATH`, `OUTPUT_CSV`, `OUTPUT_PICKLE` | dossiers `data/` et `model/` du projet |
| `creer_profil.py` | `EMBEDDINGS_PKL`, `PROFILE_DIR` | `model/embeddings.pkl` et `model/facelock/` |
| `train_svm.py` | `EMBEDDINGS_CSV`, `EMBEDDINGS_PKL`, `PROFILE_DIR` | fichiers de `model/` |
| `inference.py` | `MODEL_PATH`, `PROFIL_PATH` | `model/MobileFaceNet.tflite` + profil généré |
| `inference_knn.py` | `MODEL_PATH`, `PROFIL_PATH` | idem |
| `test_modele.py` | `MODEL_PATH` | `model/MobileFaceNet.tflite` |
| `facelock_daemon.py` | `MODEL_PATH`, `PROFILE_DIR` | idem |
| `facelock.service` | `User`, `WorkingDirectory`, `ExecStart` | ta machine Linux |

Exemple pour `generer_embeddings.py` (lignes 22-25) :

```python
DATA_DIR     = r"C:\Users\Marcos\Documents\projet_reconnaissance_fasciale\data\recup_marcos"
MODEL_PATH   = r"C:\Users\Marcos\Documents\projet_reconnaissance_fasciale\model\MobileFaceNet.tflite"
OUTPUT_CSV   = r"C:\Users\Marcos\Documents\projet_reconnaissance_fasciale\model\embeddings.csv"
OUTPUT_PICKLE= r"C:\Users\Marcos\Documents\projet_reconnaissance_fasciale\model\embeddings.pkl"
```

> Astuce : utilise des chemins **relatifs au projet** (`os.path.join(os.path.dirname(__file__), ...)`) pour ne plus jamais avoir ce problème.

---

## Étape 5 — Collecter les images (webcam)

```bash
python collecte_data.py
```

- Appuie sur **Entrée** pour démarrer, **Échap** pour arrêter.
- Les images sont sauvegardées dans `data/Marcos/` (≈ 3000 images recommandées).
- Pour une classe « autres » : relancer en changeant le nom de classe dans le script.

**Résultat attendu** : `data/Marcos/Marcos_0000.jpg`, `Marcos_0001.jpg`, …

---

## Étape 6 — Recadrer les visages avec MTCNN

```bash
python recuperation.py
```

- Détecte les visages dans `data/Marcos/` et enregistre les crops dans `data/recup_marcos/`.
- Élimine les images où aucun visage n'est détecté.

**Résultat attendu** : des fichiers `*_face_1.jpg` dans `data/recup_marcos/`.

---

## Étape 7 — Générer les embeddings

```bash
python generer_embeddings.py
```

- Charge MobileFaceNet, calcule un vecteur **192D** par image.
- Affiche un barre de progression et des statistiques (similarité moyenne/min/max).

**Résultat attendu** :
- `model/embeddings.csv` (11 Mo environ)
- `model/embeddings.pkl`

---

## Étape 8 — Créer le profil de reconnaissance

### Option A — k-NN (recommandé pour débuter)

```bash
python creer_profil.py
```

- Calcule le vecteur de référence (moyenne des embeddings), le seuil de similarité cosinus (percentile 25) et les statistiques k-NN (k=100).
- **Résultat** : `profil_complet.pkl` dans `PROFILE_DIR` (par défaut `~/.facelock/`).

### Option B — One-Class SVM

```bash
python train_svm.py
```

- Entraîne un SVM sur les embeddings, produit un rapport de classification.
- **Résultat** : `model/facelock/profil_complet.pkl`.

---

## Étape 9 — Test de reconnaissance en temps réel

```bash
# Avec le profil k-NN
python inference_knn.py

# Avec le profil SVM
python inference.py
```

- Une fenêtre webcam s'ouvre : le visage détecté est encadré et affiché **RECONNUS** ou **INCONNU**.
- Appuie sur **Échap** (ou la touche indiquée à l'écran) pour quitter.

### Étape 9 bis — Régler les seuils

Si la reconnaissance est trop stricte ou trop permissive, ajuste dans `inference_knn.py` (lignes ~43-44) :

```python
seuil_knn     = 0.75   # baisse si tu n'es jamais reconnu, monte si faux positifs
seuil_cosinus = 0.55
```

Re-teste jusqu'à obtenir un bon équilibre.

---

## Étape 10 (optionnel) — Entraîner le CNN binaire

Approche alternative (classification directe « moi / autres ») :

```bash
python train.py
```

- Charge les images de `data_recup/moi/` et `data_recup/autres/` (à créer / adapter dans le script).
- Entraîne un CNN 128×128 avec early stopping, logs TensorBoard dans `logs/`.
- Sauvegarde `modele_binaire_128.h5` + `modele_binaire_128.tflite`.

---

## Étape 11 (optionnel, Linux) — Verrouillage automatique FaceLock

1. Adapter `facelock.service` : `User`, `Group`, `WorkingDirectory`, `ExecStart`, `XAUTHORITY`.
2. Installer et activer :

```bash
sudo cp facelock.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now facelock
```

3. Suivre les logs :

```bash
journalctl -u facelock -f
```

Le daemon verrouille l'écran dès que le visage reconnu disparaît de la webcam.

---

## Dépannage

| Problème | Cause probable | Solution |
|---|---|---|
| `FileNotFoundError` | chemin absolu mal adapté | Étape 4 : vérifier chaque constante |
| `ModuleNotFoundError: mtcnn` | dépendance manquante | `pip install mtcnn` (dans le venv actif) |
| `tflite_runtime` introuvable | paquet léger absent | normal : le script retombe sur TensorFlow |
| `Erreur caméra` | webcam occupée / index faux | fermer les autres apps, changer `VideoCapture(0)` |
| « Aucune image trouvée » | `DATA_DIR` vide ou faux | vérifier le dossier (étapes 5-6) |
| Jamais reconnu | seuil trop haut | baisser `seuil_knn` / `seuil_cosinus` (étape 9 bis) |
| Reconnu à tort | seuil trop bas | monter les seuils, ajouter plus d'images (étape 5) |
| `Permission denied` (Linux, service) | chemins / droits webcam | vérifier `User`, `Group=video`, `ExecStart` |

---

## Récapitulatif de l'ordre

```
0. Prérequis
1. git clone
2. venv + pip install -r requirements.txt
3. MobileFaceNet.tflite → model/
4. Adapter les chemins dans les scripts
5. python collecte_data.py
6. python recuperation.py
7. python generer_embeddings.py
8. python creer_profil.py   (ou train_svm.py)
9. python inference_knn.py  (ou inference.py) + réglage des seuils
10. (optionnel) python train.py
11. (optionnel, Linux) facelock.service
```
