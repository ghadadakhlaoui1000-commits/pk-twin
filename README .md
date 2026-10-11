# 💊 PK-Twin — jumeau numérique pharmacocinétique

Prototype pédagogique : prédire la concentration d'un médicament dans le sang après une prise orale,
**en quelques dixièmes de milliseconde au lieu de plusieurs centaines**, avec une **incertitude** et une **alerte clinique**.

🔗 **Démo en ligne : https://pk-twin-ghada.streamlit.app**

## Principe (analogie)
Un simulateur précis mais lent (le « professeur ») corrige 600 patients virtuels. Un modèle rapide
(l'« élève » : PCA + 5 réseaux de neurones) apprend de ces corrections. Le désaccord entre les 5 réseaux
mesure l'incertitude, comme un sondage où l'on regarde si les sondeurs se contredisent.

## Fichiers
| Fichier | Rôle |
|---|---|
| `simulateur.py` | Modèle à 2 compartiments (EDO, Radau, tolérance 1e-10) + validation vs solution analytique de Bateman |
| `entrainement.py` | Hypercube latin (600 patients), PCA, ensemble de 5 MLP, évaluation, figures |
| `predire.py` | Charge le modèle rapide, renvoie courbe + incertitude |
| `app.py` | Dashboard Streamlit : curseurs, courbe ±2σ, Cmax/Tmax/AUC, alerte, sensibilité |

## Lancer
```bash
pip install -r requirements.txt
python simulateur.py        # validation (erreur ~1e-5 % vs formule exacte)
python entrainement.py      # ~3 min, crée modele_rapide.joblib et les figures
streamlit run app.py
```

## Résultats (patients de test jamais vus, mesurés sur mon ordinateur)
- Erreur L2 médiane sur les courbes : **1,2 %** (moyenne 1,9 %)
- Erreur médiane Cmax : **0,8 %**, AUC : **1,1 %**
- Accélération : **×1 759** en calcul groupé (0,4 ms par patient contre 697 ms) et **×100 à ×250** dans le dashboard, un patient à la fois ; les valeurs varient selon la machine
- Incertitude : ±2σ couvre **97 %** des points réels
- Sensibilité : la dose et la clairance dominent l'AUC 0-24 h (l'AUC est tronquée à 24 h, donc moins sensible à la clairance quand elle est faible)

## Limites (honnêtes)
Modèle pédagogique, paramètres tirés de plages plausibles, **non calibré sur données patient, non validé cliniquement**.
Prochaines étapes : doses répétées, variabilité de population (PK/PD), calibration sur données publiques.
