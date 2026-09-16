# 3D Mesh Augmentation Study

Étude de l'effet de l'augmentation de données sur la classification et la segmentation de maillages 3D, avec l'architecture **DGNet** (Jittor), sur trois datasets de référence : **Manifold10** (ModelNet10 simplifié), **SHREC16** et **Human Body Segmentation**.

## Objectif

Comparer les performances de DGNet selon 4 configurations, pour chaque dataset :

| Phase | Description |
|---|---|
| **Original** | Dataset brut, sans augmentation |
| **Augmented** | Dataset avec augmentation de données |
| **MPS** | Dataset Original, prétraité avec la pyramide multi-résolution optimisée |
| **MPS + aug** | Dataset Augmented, avec la même pyramide optimisée |

## Structure du projet

```
3d-mesh-augmentation-study/
├── adapters/          # Adaptateurs entre les datasets et DGNet (conversion de labels, pooling, etc.)
├── models/            # Sous-module Git : implémentation officielle de DGNet (li-xl/DGNet)
├── configs/           # Fichiers de configuration
├── datasets/          # Scripts/métadonnées de préparation des datasets (les données elles-mêmes sont sur Kaggle)
├── scripts/           # Notebooks Jupyter (.ipynb) — un par dataset × phase
├── resultats_dgnet.xlsx   # Tableau récapitulatif de tous les résultats (4 phases × 3 datasets)
├── cahier_des_charges.pdf # Spécification du projet
└── .gitmodules         # Déclaration du sous-module DGNet
```

## Datasets et notebooks

Chaque notebook clone ce dépôt, installe un environnement Python 3.10 isolé pour Jittor (incompatible avec le Python du kernel Kaggle), prétraite les maillages, entraîne DGNet, puis évalue le modèle.

| Dataset | Original | Augmented | MPS | MPS + aug |
|---|---|---|---|---|
| **Manifold10** | `dgnet_manifold10_original.ipynb` | `dgnet_manifold10_augmented.ipynb` | `dgnet_manifold10_mps.ipynb` | `dgnet_manifold10_mps_aug.ipynb` |
| **SHREC16** | `dgnet_shrec16_original.ipynb` | `dgnet_shrec16_augmented.ipynb` | `dgnet_shrec16_mps.ipynb` | `dgnet_shrec16_mps_aug.ipynb` |
| **Human Seg** | `dgnet_humanseg_original.ipynb` | `dgnet_humanseg_augmented.ipynb` | `dgnet_humanseg_mps.ipynb` | `dgnet_humanseg_mps_aug.ipynb` |

Manifold10 et SHREC16 sont des tâches de **classification** (10 et 30 classes respectivement) ; Human Seg est une tâche de **segmentation par face** (8 parties du corps).

## Installation

Ce projet est conçu pour tourner sur **Kaggle** (ou Google Colab), avec accélérateur GPU activé. Chaque notebook clone le dépôt automatiquement :

```bash
git clone --recurse-submodules https://github.com/Bilal51002/3d-mesh-augmentation-study.git
```

Le sous-module `models/` (DGNet) est cloné via `.gitmodules` depuis [li-xl/DGNet](https://github.com/li-xl/DGNet).

### Environnement Jittor

Jittor nécessite Python 3.10 et des dépendances CUDA/cuDNN spécifiques. Chaque notebook crée un environnement virtuel isolé et applique les correctifs nécessaires (le serveur de téléchargement CUDA/cuDNN par défaut de Jittor étant indisponible depuis les réseaux Kaggle/Colab) :
- Utilisation du `nvcc` déjà présent sur le système hôte plutôt que le téléchargement automatique de Jittor
- Installation de cuDNN 8 et cuBLAS 11 via les paquets officiels NVIDIA sur PyPI
- Désactivation des téléchargements optionnels MKL/cutlass (`use_mkl=0`, `use_cutlass=0`)

## Résultats

Voir [`resultats_dgnet.xlsx`](./resultats_dgnet.xlsx) pour le tableau complet (accuracy, precision, recall, F1-score macro/weighted, nombre d'epochs) des 12 configurations (3 datasets × 4 phases).

Chaque notebook génère également un rapport Markdown détaillé (`RAPPORT_DGNet_*.md`) dans son dossier de sortie, incluant la configuration d'entraînement complète, la matrice de confusion, et une analyse des confusions les plus fréquentes.

## Architecture

**DGNet** — réseau de neurones sur graphe de maillage, avec pooling hiérarchique multi-résolution.
Dépôt officiel : https://github.com/li-xl/DGNet

## Cahier des charges

Voir [`cahier_des_charges.pdf`](./cahier_des_charges.pdf) pour la spécification complète du projet.
