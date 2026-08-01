# Robustesse de l'entraînement — train_robuste.py / test_robuste.py

## Problème rencontré
L'algorithme de pooling de HalfedgeCNN (half_edge_mesh_pool.py) utilise une
file de priorité (heap) pour fusionner itérativement des paires de
half-edges jusqu'à atteindre une résolution cible (pool_res). Chaque
itération retire un élément du tas, y compris lorsque la fusion candidate
est invalide (condition de connectivité non respectée). Sur des meshes à
topologie irrégulière (résultant de la réparation/décimation de CAD models
bruts de ModelNet10), le tas peut se vider avant d'atteindre la cible,
provoquant un crash (IndexError: index out of range).

## Constat important
Un diagnostic exhaustif, testant individuellement chaque mesh du dataset
(chargement + forward + backward, un par un) n'a détecté aucune erreur.
Le crash n'apparaît que lors de l'entraînement réel par batchs mélangés
(batch_size > 1, shuffle activé) : le comportement de l'algorithme dépend
des valeurs numériques exactes des features à un instant donné, qui varient
légèrement selon l'ordre de traitement des données et l'initialisation des
poids. Il s'agit d'une fragilité connue des algorithmes de pooling glouton
basés sur un tas, face à des données à topologie hétérogène, et non d'un
fichier corrompu identifiable a priori.

## Solution retenue
Modification de train.py et test.py (copies renommées train_robuste.py et
test_robuste.py) : chaque batch est protégé par un bloc try/except. Un
batch qui échoue est ignoré et loggé (batchs_ignores.log,
batchs_test_ignores.log), sans interrompre l'entraînement.

## Impact mesuré
Sur le test à 1 epoch : 3/866 batchs d'entraînement ignorés (0.35%),
1/195 batchs de test ignoré (0.5%). Impact négligeable sur le volume de
données effectivement utilisé.

## Fichiers modifiés (copies du repo officiel)
- train.py -> train_robuste.py
- test.py -> test_robuste.py
Fichiers originaux conservés intacts dans le repo officiel cloné pour
traçabilité (section 5.1 du cahier des charges).
