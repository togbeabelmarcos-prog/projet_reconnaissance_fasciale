"""
Entraîne un One-Class SVM sur les embeddings de "moi"
Split 80% train / 20% test
Évalue les performances et sauvegarde le modèle

Peut charger depuis CSV ou Pickle
"""
import os
import pickle
import csv
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────
EMBEDDINGS_CSV  = r"C:\Users\Marcos\Documents\data\model\embeddings.csv"
PROFILE_DIR     = os.path.expanduser(r"C:\Users\Marcos\Documents\data\model\facelock")
PROFIL_FINAL    = os.path.join(PROFILE_DIR, "profil_complet.pkl")

# ──────────────────────────────────────────
# CHARGEMENT DES EMBEDDINGS
# ──────────────────────────────────────────
print(" Chargement des embeddings...")

# Essayer CSV en premier, sinon pickle
if os.path.exists(EMBEDDINGS_CSV):
    print(f"    Source : {EMBEDDINGS_CSV}")
    embeddings = []
    images = []
    with open(EMBEDDINGS_CSV, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # sauter l'en-tête
        for row in reader:
            images.append(row[0])
            emb = [float(x) for x in row[1:]]
            embeddings.append(emb)
    embeddings = np.array(embeddings)
    n_total = len(embeddings)
    print(f"    {n_total} embeddings chargés depuis CSV")
    print(f"   Dimension : {embeddings.shape[1]}D\n")
elif os.path.exists(EMBEDDINGS_PKL):
    print(f"    Source : {EMBEDDINGS_PKL}")
    with open(EMBEDDINGS_PKL, 'rb') as f:
        data = pickle.load(f)
    embeddings = np.array(data['embeddings'])
    images = data['images']
    n_total = len(embeddings)
    print(f"   {n_total} embeddings chargés depuis Pickle")
    print(f"   Dimension : {data['embedding_dim']}D\n")
else:
    print(" Aucun fichier d'embeddings trouvé !")
    print("   Lance d'abord : python generer_embeddings.py")
    exit(1)

# ──────────────────────────────────────────
# SPLIT 80% TRAIN / 20% TEST
# ──────────────────────────────────────────
print(" Split 80% train / 20% test...")
X_train, X_test, img_train, img_test = train_test_split(
    embeddings, images, test_size=0.2, random_state=42
)

print(f"   Train : {len(X_train)} embeddings")
print(f"   Test  : {len(X_test)} embeddings\n")

# ──────────────────────────────────────────
# NORMALISATION
# ──────────────────────────────────────────
print(" Normalisation des données...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Normalisé\n")

# ──────────────────────────────────────────
# ENTRAÎNEMENT DU SVM
# ──────────────────────────────────────────
print("️  Entraînement One-Class SVM...")
print("   Cela peut prendre quelques secondes...\n")

# One-Class SVM : apprend la frontière de ton cluster d'embeddings
# nu = proportion attendue d'outliers (ici ~5% pour être strict)
svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm.fit(X_train_scaled)

print(" SVM entraîné\n")

# ──────────────────────────────────────────
# ÉVALUATION SUR LE TRAIN
# ──────────────────────────────────────────
print(" Évaluation sur le TRAIN...")
y_train_pred = svm.predict(X_train_scaled)
train_acc = np.sum(y_train_pred == 1) / len(y_train_pred) * 100
print(f"   Taux reconnaissance train : {train_acc:.1f}%")

# ──────────────────────────────────────────
# ÉVALUATION SUR LE TEST
# ──────────────────────────────────────────
print("\n Évaluation sur le TEST...")
y_test_pred = svm.predict(X_test_scaled)
test_acc = np.sum(y_test_pred == 1) / len(y_test_pred) * 100
print(f"   Taux reconnaissance test : {test_acc:.1f}%")

# Matrice de confusion (sur test)
print("\n   Matrice de confusion (test) :")
cm = confusion_matrix(y_test_pred, [1]*len(y_test_pred))
print(f"      Prédit")
print(f"      1     -1")
print(f"Réel 1  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
if len(cm) > 1:
    print(f"     -1  [{cm[1][0]:4d}  {cm[1][1]:4d}]")
else:
    print(f"     -1  [   0     0]")

# ──────────────────────────────────────────
# CALCUL DU SEUIL COSINUS
# ──────────────────────────────────────────
print("\n Calcul du seuil cosinus...")

# Vecteur de référence = moyenne des embeddings du train
vecteur_reference = np.mean(X_train, axis=0)
vecteur_reference = vecteur_reference / np.linalg.norm(vecteur_reference)

# Similarités de tous les embeddings (train + test) par rapport au vecteur de référence
all_embeddings = np.array(embeddings)
similarites = [float(np.dot(vecteur_reference, emb)) for emb in all_embeddings]

# Seuil = percentile 5 (95% de tes embeddings passent)
seuil_cosinus = np.percentile(similarites, 5)

print(f"   Similarité moyenne : {np.mean(similarites):.4f}")
print(f"   Similarité min     : {np.min(similarites):.4f}")
print(f"   Similarité max     : {np.max(similarites):.4f}")
print(f"   Seuil cosinus      : {seuil_cosinus:.4f} (percentile 5)")

# Vérification : combien passent le seuil cosinus ?
n_pass_cosinus = sum(1 for s in similarites if s >= seuil_cosinus)
print(f"   Embeddings passant le seuil : {n_pass_cosinus}/{len(similarites)} ({n_pass_cosinus/len(similarites)*100:.1f}%)")

# ──────────────────────────────────────────
# TEST DE COHÉRENCE C1 (cosinus) + C2 (SVM)
# ──────────────────────────────────────────
print("\n Test de cohérence C1 (cosinus) + C2 (SVM)...")
desaccords = 0
for i, emb in enumerate(all_embeddings):
    # Test cosinus
    sim = float(np.dot(vecteur_reference, emb))
    c1_ok = sim >= seuil_cosinus
    
    # Test SVM
    emb_scaled = scaler.transform([emb])
    c2_ok = svm.predict(emb_scaled)[0] == 1
    
    if c1_ok != c2_ok:
        desaccords += 1

taux_desaccord = desaccords / len(all_embeddings) * 100
print(f"   Désaccords C1 ↔ C2 : {desaccords}/{len(all_embeddings)} ({taux_desaccord:.1f}%)")

if taux_desaccord < 5:
    print("    Excellente cohérence")
elif taux_desaccord < 15:
    print("     Cohérence acceptable")
else:
    print("    Mauvaise cohérence — à améliorer")

# ──────────────────────────────────────────
# SAUVEGARDE DU PROFIL
# ──────────────────────────────────────────
print(f"\n Sauvegarde du profil...")
os.makedirs(PROFILE_DIR, exist_ok=True)

profil = {
    'vecteur_reference': vecteur_reference,
    'seuil_cosinus': seuil_cosinus,
    'svm_model': svm,
    'svm_scaler': scaler,
    'n_samples_bruts': len(X_train),
    'n_samples_total': len(all_embeddings),
    'tous_embeddings': embeddings,
    'taux_desaccord': taux_desaccord,
    'modele': 'MobileFaceNet 192D + One-Class SVM',
    'train_acc': train_acc,
    'test_acc': test_acc,
    'images_train': img_train,
    'images_test': img_test
}

with open(PROFIL_FINAL, 'wb') as f:
    pickle.dump(profil, f)

os.chmod(PROFIL_FINAL, 0o600)

print(f" Profil sauvegardé → {PROFIL_FINAL}")

# ──────────────────────────────────────────
# BILAN FINAL
# ──────────────────────────────────────────
print("\n" + "="*60)
print("   BILAN FINAL")
print("="*60)
print(f"  Embeddings totaux    : {n_total}")
print(f"  Train                : {len(X_train)} ({len(X_train)/n_total*100:.0f}%)")
print(f"  Test                 : {len(X_test)} ({len(X_test)/n_total*100:.0f}%)")
print(f"  Accuracy Train       : {train_acc:.1f}%")
print(f"  Accuracy Test        : {test_acc:.1f}%")
print(f"  Seuil cosinus        : {seuil_cosinus:.4f}")
print(f"  Cohérence C1 ↔ C2    : {100-taux_desaccord:.1f}%")
print("="*60)

if train_acc > 95 and test_acc > 90:
    print("\n   MODÈLE PRÊT POUR L'INFÉRENCE !")
    print("  ️  Lance : python inference.py")
elif train_acc > 90 and test_acc > 85:
    print("\n  ️  Modèle acceptable mais perfectible")
    print("  ️  Tu peux lancer : python inference.py")
    print("  Ou améliorer avec plus de données variées")
else:
    print("\n  ️  Performances insuffisantes")
    print("  ️  Vérifie la qualité de tes images dans /data_recup/moi/")
    print("  ️  Ajoute plus de variété (angles, éclairages)")
