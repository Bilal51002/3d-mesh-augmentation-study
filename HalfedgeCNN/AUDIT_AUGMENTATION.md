# Audit de l'augmentation interne — HalfedgeCNN

## Constat
Le repo officiel applique de l'augmentation interne activée par défaut
dans `scripts/settings/classification_training_settings.txt` :
- `number_augmentations = 20`
- `flip_edges = 0.2`
- `slide_verts = 0.2`

Mécanismes implémentés dans `models/layers/half_edge_mesh_prepare.py` :
- `scale_verts()` : mise à l'échelle aléatoire des sommets (désactivé par défaut, non concerné)
- `flip_edges()` : retournement aléatoire d'arêtes (angle dièdre > 155°)
- `slide_verts()` : déplacement aléatoire de sommets sur la surface du mesh

Dans le code source (`options/train_options.py`), ces paramètres sont désactivés par
défaut (0 ou False). C'est le fichier de configuration `classification_training_settings.txt`
qui les active pour un entraînement standard.

## Décision
Désactivation complète pour tous les runs (dataset original ET dataset augmenté externe),
afin d'isoler l'effet de l'augmentation externe fournie par l'encadrant, sans interférence
de l'augmentation interne du repo.

## Modification appliquée
Création de `scripts/settings/classification_training_settings_no_aug.txt` avec :
- `number_augmentations = 1`
- `flip_edges = 0`
- `slide_verts = 0`

Le fichier original `classification_training_settings.txt` est conservé intact pour
traçabilité et documentation de l'implémentation officielle utilisée (section 5.1).

## Commit de référence
Repo officiel : https://github.com/IngmarLudwig/HalfedgeCNN
Commit utilisé : a958b67dcd026f6a5cfa027c6f146b3f60ee293d (24 octobre 2024)
