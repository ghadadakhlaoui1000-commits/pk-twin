"""
predire.py — utilise le modèle rapide déjà entraîné (c'est lui qu'appelle le dashboard).
Renvoie la courbe moyenne ET l'incertitude (écart-type entre les 5 réseaux).
"""
import numpy as np
import joblib

_M = None


def _charger():
    global _M
    if _M is None:
        _M = joblib.load("modele_rapide.joblib")
    return _M


def predire(X):
    """X : tableau (n, 6) [dose, ka, CL, V1, Q, V2] -> (courbes moyennes, écarts-types)."""
    M = _charger()
    X = np.atleast_2d(X)
    xin = M["sx"].transform(np.log(X))
    courbes = []
    for net in M["membres"]:
        z = M["sz"].inverse_transform(net.predict(xin))
        courbes.append(np.clip(M["pca"].inverse_transform(z), 0, None) ** 2)
    courbes = np.array(courbes)
    return courbes.mean(axis=0), courbes.std(axis=0)
