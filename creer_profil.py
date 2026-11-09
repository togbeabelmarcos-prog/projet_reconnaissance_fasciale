"""
Crée le profil de reconnaissance avec approche k-NN (sans SVM)
Calcule le seuil de similarité basé sur les embeddings du dataset
"""
import os
import pickle
import numpy as np

# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────
EMBEDDINGS_PKL  = "/home/nicaise/Bureau/cyber_agent/projet licence/embeddings_moi.pkl"
PROFILE_DIR     = os.path.expanduser("~/.facelock")
PROFIL_FINAL    = os.path.join(PROFILE_DIR, "profil_complet.pkl")

# ──────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────
print(" Chargement des embeddings...")
with open(EMBEDDINGS_PKL, 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']
n_total = len(embeddings)
dim = data['embedding_dim']

print(f" {n_total} embeddings chargés ({dim}D)\n")

# ──────────────────────────────────────────
# VECTEUR DE RÉFÉRENCE (moyenne des embeddings)
# ──────────────────────────────────────────
print(" Calcul du vecteur de référence...")
vecteur_reference = np.mean(embeddings, axis=0)
vecteur_reference = vecteur_reference / np.linalg.norm(vecteur_reference)
print(f"    Vecteur de référence calculé\n")

# ──────────────────────────────────────────
# CALCUL DES SIMILARITÉS
# ──────────────────────────────────────────
print(" Calcul des similarités avec le vecteur de référence...")
similarites = [float(np.dot(vecteur_reference, emb)) for emb in embeddings]

print(f"   Similarité moyenne : {np.mean(similarites):.4f}")
print(f"   Similarité min     : {np.min(similarites):.4f}")
print(f"   Similarité max     : {np.max(similarites):.4f}")

# ──────────────────────────────────────────
# CALCUL DU SEUIL (percentile 25 pour être plus permissif)
# ──────────────────────────────────────────
print("\n Calcul du seuil de reconnaissance...")
seuil_cosinus = np.percentile(similarites, 25)  # 75% de tes embeddings passent
print(f"   Seuil cosinus      : {seuil_cosinus:.4f} (percentile 25)")

n_pass = sum(1 for s in similarites if s >= seuil_cosinus)
print(f"   Embeddings passant : {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%)")

# ──────────────────────────────────────────
# STATISTIQUES K-NN
# ──────────────────────────────────────────
print("\n Statistiques k-NN (k=100)...")
k = min(100, n_total)

# Pour chaque embedding, calculer la similarité moyenne avec ses k-1 plus proches
knn_moyennes = []
for i, emb in enumerate(embeddings):
    # Similarités avec tous les autres
    sims = [np.dot(emb, embeddings[j]) for j in range(n_total) if j != i]
    sims.sort(reverse=True)
    knn_moy = np.mean(sims[:k-1])  # k-1 plus proches
    knn_moyennes.append(knn_moy)

print(f"   k-NN moyen (k={k})   : {np.mean(knn_moyennes):.4f}")
print(f"   k-NN min             : {np.min(knn_moyennes):.4f}")
print(f"   k-NN max             : {np.max(knn_moyennes):.4f}")

# Seuil k-NN (percentile 25 aussi)
seuil_knn = np.percentile(knn_moyennes, 25)
print(f"   Seuil k-NN           : {seuil_knn:.4f} (percentile 25)")

# ──────────────────────────────────────────
# SAUVEGARDE DU PROFIL
# ──────────────────────────────────────────
print(f"\n Sauvegarde du profil...")
os.makedirs(PROFILE_DIR, exist_ok=True)

profil = {
    'vecteur_reference': vecteur_reference,
    'seuil_cosinus': seuil_cosinus,
    'seuil_knn': seuil_knn,
    'k': k,
    'tous_embeddings': embeddings,
    'images': data['images'],
    'n_embeddings': n_total,
    'modele': 'MobileFaceNet 192D + k-NN',
    'knn_moyennes_stats': {
        'mean': np.mean(knn_moyennes),
        'min': np.min(knn_moyennes),
        'max': np.max(knn_moyennes)
    }
}

with open(PROFIL_FINAL, 'wb') as f:
    pickle.dump(profil, f)

os.chmod(PROFIL_FINAL, 0o600)

print(f" Profil sauvegardé → {PROFIL_FINAL}")

# ──────────────────────────────────────────
# BILAN
# ──────────────────────────────────────────
print("\n" + "="*60)
print("   BILAN")
print("="*60)
print(f"  Embeddings totaux    : {n_total}")
print(f"  Dimension            : {dim}D")
print(f"  Seuil cosinus        : {seuil_cosinus:.4f}")
print(f"  Seuil k-NN (k={k})    : {seuil_knn:.4f}")
print(f"  k-NN moyen           : {np.mean(knn_moyennes):.4f}")
print("="*60)

if np.mean(knn_moyennes) > 0.7:
    print("\n   Tes embeddings sont TRÈS COHÉRENTS")
    print("    Lance : python inference_knn.py")
elif np.mean(knn_moyennes) > 0.5:
    print("\n    Embeddings moyennement cohérents")
    print("    Lance : python inference_knn.py et teste")
else:
    print("\n    Embeddings peu cohérents")
    print("    Vérifie la qualité de ton dataset")
