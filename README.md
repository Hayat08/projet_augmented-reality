# Augmented Reality Project 
# Vision augmentée et recalage dynamique 2D multi-objets en temps réel

## 1. Présentation du projet

Ce projet consiste à développer une application de réalité augmentée en C++ avec OpenCV. Elle permet de détecter plusieurs images de référence dans un flux vidéo provenant d'une webcam, de suivre leurs déplacements et d'effectuer un recalage vidéo afin d'intégrer du contenu virtuel dans la scène.

## 2. Objectifs

- Détecter plusieurs images de référence simultanément.
- Localiser chaque image et déterminer ses quatre coins.
- Suivre les déplacements des images en temps réel.
- Calculer les transformations géométriques nécessaires au recalage.
- Incruster un contenu vidéo sur les images détectées.

## 3. Technologies et méthodes utilisées

- **Langage :** C++
- **Bibliothèque :** OpenCV
- **ORB / AKAZE :** extraction des caractéristiques.
- **BFMatcher :** mise en correspondance des descripteurs.
- **RANSAC et Homographie :** estimation de la transformation géométrique.
- **Optical Flow (Lucas-Kanade) :** suivi des images entre les frames.
- **Perspective Transform :** projection et recalage du contenu vidéo.

## 4. Organisation du projet

```text
augmented_reality_project/
│
├── main.cpp
├── Reference Images-20261114/
│   └── Images de référence
│
└── Reference Videos-20261114/
    └── Videos de référence
```

## 5. Fonctionnement général

Le projet est organisé en deux parties principales :

**Partie 1 – Détection et tracking :**
- Chargement des images de référence.
- Détection des caractéristiques avec ORB ou AKAZE.
- Recherche des correspondances dans les frames de la webcam.
- Calcul de l'homographie et localisation des images.
- Suivi des images détectées avec l'Optical Flow.

**Partie 2 – Recalage vidéo :**
- Exploitation des informations de détection et des coordonnées des images.
- Calcul de la transformation nécessaire au recalage.
- Adaptation de la vidéo à la perspective des images détectées.
- Incrustation du contenu vidéo dans la scène.

## 6. Exécution

**Prérequis :**
- Visual Studio
- C++ et OpenCV correctement configurés.
- Une webcam.
- Les images de référence nécessaires au projet.

Ouvrir le projet dans Visual Studio, vérifier la configuration des chemins, puis compiler et exécuter l'application.

## 7. Résultat attendu

Obtenir une application de réalité augmentée capable de détecter et suivre plusieurs images simultanément, puis d'incruster un contenu vidéo adapté à leur position et à leur perspective dans le flux de la webcam.
