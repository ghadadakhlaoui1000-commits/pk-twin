"""
simulateur.py — le "PROFESSEUR" (modèle précis mais lent)

Rôle : calculer la concentration d'un médicament dans le sang au cours du temps
après une prise orale, avec un modèle pharmacocinétique à 2 compartiments.

Analogie : le corps est un système de 3 réservoirs d'eau reliés par des tuyaux :
  - l'intestin (où le comprimé se dissout)      -> a_gut
  - le sang (compartiment central)              -> a_c
  - les tissus (compartiment périphérique)      -> a_p
Le médicament coule de l'intestin vers le sang (ka), va et vient entre sang et
tissus (Q), et est éliminé par le foie/reins (CL).
"""
import time
import numpy as np
from scipy.integrate import solve_ivp

# Les 6 paramètres d'entrée et leurs plages réalistes (min, max)
PARAMS = {
    "dose": (50.0, 1000.0),   # mg
    "ka":   (0.3, 3.0),       # 1/h  vitesse d'absorption
    "CL":   (2.0, 20.0),      # L/h  clairance (élimination)
    "V1":   (15.0, 80.0),     # L    volume du sang/compartiment central
    "Q":    (1.0, 15.0),      # L/h  échange sang <-> tissus
    "V2":   (20.0, 150.0),    # L    volume des tissus
}
NOMS = list(PARAMS.keys())
T_FIN = 24.0                                  # heures simulées
TEMPS = np.linspace(0.0, T_FIN, 121)          # 121 points (toutes les 12 min)


def equations(t, y, ka, CL, V1, Q, V2):
    a_gut, a_c, a_p = y
    d_gut = -ka * a_gut
    d_c = ka * a_gut - (CL / V1) * a_c - (Q / V1) * a_c + (Q / V2) * a_p
    d_p = (Q / V1) * a_c - (Q / V2) * a_p
    return [d_gut, d_c, d_p]


def simuler_haute_fidelite(dose, ka, CL, V1, Q, V2):
    """Résolution numérique très précise (tolérance serrée = lent mais exact)."""
    sol = solve_ivp(equations, (0, T_FIN), [dose, 0.0, 0.0],
                    args=(ka, CL, V1, Q, V2), t_eval=TEMPS,
                    method="Radau", rtol=1e-10, atol=1e-12)
    return sol.y[1] / V1                      # concentration sang en mg/L


def solution_analytique_1cpt(dose, ka, CL, V1, t=TEMPS):
    """Formule exacte (Bateman) du modèle à 1 compartiment, sert à VALIDER le simulateur
    (équivalent de la solution de Lamé dans le projet de référence)."""
    ke = CL / V1
    if abs(ka - ke) < 1e-9:
        ka = ka + 1e-6
    return dose * ka / (V1 * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def indicateurs(conc):
    """Résume la courbe en chiffres cliniques : Cmax, Tmax, AUC."""
    i = int(np.argmax(conc))
    return {"Cmax": float(conc[i]), "Tmax": float(TEMPS[i]),
            "AUC": float(np.trapezoid(conc, TEMPS))}


def mesurer_temps(n=20):
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(n):
        p = [rng.uniform(*PARAMS[k]) for k in NOMS]
        simuler_haute_fidelite(*p)
    return (time.perf_counter() - t0) / n


if __name__ == "__main__":
    # Validation : avec Q très petit, le modèle 2 compartiments ≈ 1 compartiment
    p = dict(dose=500, ka=1.2, CL=8.0, V1=40.0, Q=1e-6, V2=50.0)
    num = simuler_haute_fidelite(**p)
    ana = solution_analytique_1cpt(p["dose"], p["ka"], p["CL"], p["V1"])
    err = np.max(np.abs(num - ana)) / np.max(ana) * 100
    print(f"Validation vs solution analytique : erreur max = {err:.6f} %")
    print(f"Temps moyen par simulation : {mesurer_temps()*1000:.1f} ms")
