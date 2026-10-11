"""
entrainement.py — l'"ÉLÈVE" (modèle rapide qui imite le professeur)

Étapes :
 1. On tire 600 patients virtuels (hypercube latin = couverture régulière des cas)
 2. Le professeur (simulateur.py) calcule leur courbe -> c'est le "cahier de corrections"
 3. On compresse les courbes avec une PCA (= POD-ROM : garder seulement les formes essentielles)
 4. Un ensemble de 5 petits réseaux de neurones apprend à prédire ces formes
    -> le désaccord entre les 5 réseaux donne l'INCERTITUDE (ce que le projet de référence n'a pas)
 5. On mesure erreur et vitesse, et on trace les figures
"""
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from simulateur import PARAMS, NOMS, TEMPS, simuler_haute_fidelite, indicateurs

N_SIM, N_MEMBRES, N_COMP = 600, 5, 8
rng = np.random.default_rng(42)


def echantillonner(n, seed):
    """Hypercube latin dans l'espace des paramètres (échelle log = plus naturel)."""
    u = qmc.LatinHypercube(d=len(NOMS), seed=seed).random(n)
    lo = np.log([PARAMS[k][0] for k in NOMS]); hi = np.log([PARAMS[k][1] for k in NOMS])
    return np.exp(lo + u * (hi - lo))


def vers_entree(X):
    return np.log(X)


if __name__ == "__main__":
    # --- 1-2. Génération des données par le professeur ---
    import os
    if os.path.exists("donnees.npz"):
        d = np.load("donnees.npz"); X, Y = d["X"], d["Y"]
    else:
        X = echantillonner(N_SIM, seed=1)
        t0 = time.perf_counter()
        Y = np.array(Parallel(n_jobs=-1)(delayed(simuler_haute_fidelite)(*x) for x in X))
        print(f"{N_SIM} simulations en {time.perf_counter()-t0:.1f} s")
        np.savez("donnees.npz", X=X, Y=Y)

    idx = rng.permutation(N_SIM); n_tr = int(0.8 * N_SIM)
    tr, te = idx[:n_tr], idx[n_tr:]

    # --- 3. Compression PCA (on travaille sur sqrt pour mieux représenter les faibles concentrations) ---
    Ys = np.sqrt(np.clip(Y, 0, None))
    pca = PCA(n_components=N_COMP).fit(Ys[tr])
    Z = pca.transform(Ys)
    print(f"PCA : {N_COMP} composantes expliquent {pca.explained_variance_ratio_.sum()*100:.4f} % de la variance")
    sx, sz = StandardScaler().fit(vers_entree(X[tr])), StandardScaler().fit(Z[tr])

    # --- 4. Ensemble de réseaux de neurones ---
    membres = []
    for m in range(N_MEMBRES):
        boot = rng.choice(tr, size=len(tr), replace=True)   # chaque élève voit un échantillon un peu différent
        net = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs",
                           max_iter=5000, tol=1e-10, alpha=1e-6, random_state=m)
        net.fit(sx.transform(vers_entree(X[boot])), sz.transform(Z[boot]))
        membres.append(net)
    joblib.dump({"pca": pca, "sx": sx, "sz": sz, "membres": membres}, "modele_rapide.joblib")

    # --- 5. Évaluation sur patients jamais vus ---
    from predire import predire
    t0 = time.perf_counter(); moy, std = predire(X[te]); t_rapide = (time.perf_counter() - t0) / len(te)
    err_l2 = np.linalg.norm(moy - Y[te], axis=1) / np.linalg.norm(Y[te], axis=1) * 100
    ind_v = np.array([[indicateurs(c)[k] for k in ("Cmax", "AUC")] for c in Y[te]])
    ind_p = np.array([[indicateurs(c)[k] for k in ("Cmax", "AUC")] for c in moy])
    err_ind = np.abs(ind_p - ind_v) / ind_v * 100
    t_lent = np.mean([ (lambda s: (simuler_haute_fidelite(*X[i]), time.perf_counter()-s)[1])(time.perf_counter()) for i in te[:15]])
    # couverture de l'incertitude : % de points réels dans moy ± 2*std
    couv = np.mean(np.abs(moy - Y[te]) <= 2 * std + 1e-3) * 100
    print(f"Erreur L2 médiane sur les courbes : {np.median(err_l2):.2f} %  (moyenne {err_l2.mean():.2f} %)")
    print(f"Erreur médiane Cmax : {np.median(err_ind[:,0]):.2f} %  | AUC : {np.median(err_ind[:,1]):.2f} %")
    print(f"Temps : professeur {t_lent*1000:.1f} ms vs élève {t_rapide*1000:.3f} ms -> x{t_lent/t_rapide:.0f}")
    print(f"Couverture de l'incertitude (±2σ) : {couv:.1f} % des points")

    # --- Figures ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(err_l2, bins=25, color="#c0392b", alpha=.7); ax[0].set_xlabel("erreur L2 relative (%)"); ax[0].set_title("Erreur sur patients test")
    for j, k in enumerate(["Cmax (mg/L)", "AUC (mg·h/L)"]):
        pass
    ax[1].scatter(ind_v[:, 0], ind_p[:, 0], s=10); lim = [0, ind_v[:, 0].max()]; ax[1].plot(lim, lim, "k--")
    ax[1].set_xlabel("Cmax simulé"); ax[1].set_ylabel("Cmax prédit"); ax[1].set_title("Cmax")
    ax[2].scatter(ind_v[:, 1], ind_p[:, 1], s=10, color="#16a085"); lim = [0, ind_v[:, 1].max()]; ax[2].plot(lim, lim, "k--")
    ax[2].set_xlabel("AUC simulée"); ax[2].set_ylabel("AUC prédite"); ax[2].set_title("AUC")
    plt.tight_layout(); plt.savefig("fig_erreurs.png", dpi=150); plt.close()

    i = te[0]; m1, s1 = predire(X[i:i+1])
    plt.figure(figsize=(7, 4)); plt.plot(TEMPS, Y[i], "b", lw=2, label="Professeur (précis)")
    plt.plot(TEMPS, m1[0], "r--", label="Élève (rapide)")
    plt.fill_between(TEMPS, m1[0]-2*s1[0], m1[0]+2*s1[0], color="r", alpha=.2, label="Incertitude ±2σ")
    plt.xlabel("temps (h)"); plt.ylabel("concentration (mg/L)"); plt.legend(); plt.tight_layout()
    plt.savefig("fig_courbe.png", dpi=150); plt.close()
