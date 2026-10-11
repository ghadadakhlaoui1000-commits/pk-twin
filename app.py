"""
app.py — le DASHBOARD Streamlit (la vitrine du projet)

Le médecin/étudiant bouge 6 curseurs -> le modèle rapide répond instantanément avec
la courbe, l'incertitude, les indicateurs et une alerte si on sort de la fenêtre thérapeutique.
"""
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from simulateur import PARAMS, NOMS, TEMPS, simuler_haute_fidelite, indicateurs
from predire import predire

st.set_page_config(page_title="PK-Twin", page_icon="💊", layout="wide")
st.title("💊 PK-Twin — jumeau numérique pharmacocinétique")
st.caption("Modèle pédagogique à 2 compartiments, accéléré par PCA + ensemble de réseaux de neurones. "
           "Non calibré sur données patient, non validé cliniquement.")

PROFILS = {
    "A — Adulte standard":   dict(dose=500, ka=1.2, CL=8.0,  V1=40.0, Q=5.0,  V2=60.0),
    "B — Insuffisance rénale (CL ↓)": dict(dose=500, ka=1.2, CL=3.0, V1=40.0, Q=5.0, V2=60.0),
    "C — Absorption lente":  dict(dose=500, ka=0.4, CL=8.0,  V1=40.0, Q=5.0,  V2=60.0),
    "D — Gros volume de distribution": dict(dose=500, ka=1.2, CL=8.0, V1=75.0, Q=10.0, V2=140.0),
}
UNITES = {"dose": "mg", "ka": "1/h", "CL": "L/h", "V1": "L", "Q": "L/h", "V2": "L"}

with st.sidebar:
    st.header("Patient virtuel")
    profil = st.selectbox("Profil", list(PROFILS))
    base = PROFILS[profil]
    vals = {}
    for k in NOMS:
        lo, hi = PARAMS[k]
        vals[k] = st.slider(f"{k} ({UNITES[k]})", float(lo), float(hi), float(base[k]), key=f"{profil}-{k}")
    st.header("Fenêtre thérapeutique")
    cmin = st.number_input("Concentration minimale efficace (mg/L)", 0.0, 50.0, 2.0)
    cmax_tox = st.number_input("Seuil toxique (mg/L)", 0.1, 100.0, 12.0)
    comparer = st.checkbox("Comparer au professeur (lent)", value=True)

x = np.array([[vals[k] for k in NOMS]])
predire(x)  # chauffe : le 1er appel charge le modèle depuis le disque, on ne le chronomètre pas
N_REP = 50
t0 = time.perf_counter()
for _ in range(N_REP):
    moy, std = predire(x)
t_rapide = (time.perf_counter() - t0) / N_REP   # temps moyen d'UNE prédiction
moy, std = moy[0], std[0]
ind = indicateurs(moy)
haut, bas = moy + 2 * std, np.clip(moy - 2 * std, 0, None)

# --- Alerte clinique ---
if haut.max() > cmax_tox and moy.max() > cmax_tox:
    st.error(f"⚠️ Risque de toxicité : Cmax prévu {moy.max():.1f} mg/L > seuil {cmax_tox} mg/L")
elif haut.max() > cmax_tox:
    st.warning(f"Zone d'incertitude : le Cmax pourrait dépasser le seuil toxique (jusqu'à {haut.max():.1f} mg/L)")
elif moy.max() < cmin:
    st.warning(f"Sous-dosage probable : Cmax prévu {moy.max():.1f} mg/L < minimum efficace {cmin} mg/L")
else:
    st.success("Concentration dans la fenêtre thérapeutique")

col1, col2 = st.columns([3, 2])
with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(cmin, cmax_tox, color="green", alpha=.08, label="Fenêtre thérapeutique")
    ax.plot(TEMPS, moy, "r-", lw=2, label="Modèle rapide")
    ax.fill_between(TEMPS, bas, haut, color="r", alpha=.2, label="Incertitude ±2σ")
    if comparer:
        ref = simuler_haute_fidelite(*x[0]); ax.plot(TEMPS, ref, "b--", label="Haute fidélité")
    ax.set_xlabel("temps (h)"); ax.set_ylabel("concentration (mg/L)"); ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=.3)
    st.pyplot(fig)

with col2:
    st.subheader("Indicateurs")
    st.metric("Cmax", f"{ind['Cmax']:.2f} mg/L", help="Pic de concentration")
    st.metric("Tmax", f"{ind['Tmax']:.1f} h", help="Moment du pic")
    st.metric("AUC 0-24h", f"{ind['AUC']:.1f} mg·h/L", help="Exposition totale au médicament")
    st.metric("Incertitude max", f"± {2*std.max():.2f} mg/L")
    st.caption(f"⚡ Calcul rapide : {t_rapide*1000:.2f} ms")
    if comparer:
        t0 = time.perf_counter(); ref = simuler_haute_fidelite(*x[0]); t_lent = time.perf_counter() - t0
        ri = indicateurs(ref)
        st.caption(f"🐢 Haute fidélité : {t_lent*1000:.0f} ms (×{t_lent/t_rapide:.0f} plus lent)")
        st.caption(f"Écart Cmax : {abs(ind['Cmax']-ri['Cmax'])/ri['Cmax']*100:.2f} %  |  "
                   f"AUC : {abs(ind['AUC']-ri['AUC'])/ri['AUC']*100:.2f} %")

# --- Analyse de sensibilité : quel paramètre compte le plus ? ---
st.subheader("Quel paramètre influence le plus l'AUC ?")
st.caption("On fait varier chaque paramètre de ±20 % autour du patient actuel, les autres restant fixes.")
effets = {}
for i, k in enumerate(NOMS):
    res = []
    for f in (0.8, 1.2):
        xx = x.copy(); xx[0, i] = np.clip(xx[0, i] * f, *PARAMS[k])
        res.append(indicateurs(predire(xx)[0][0])["AUC"])
    effets[k] = abs(res[1] - res[0]) / max(ind["AUC"], 1e-9) * 100
fig2, ax2 = plt.subplots(figsize=(7, 2.8))
ordre = sorted(effets, key=effets.get)
ax2.barh(ordre, [effets[k] for k in ordre], color="#16a085")
ax2.set_xlabel("variation de l'AUC (%) pour ±20 % du paramètre"); st.pyplot(fig2)