# Resultats — HalfedgeCNN sur ModelNet10 (dataset original, sans augmentation)

## Configuration
number_input_faces=500, pool_res=1200/900/600/360, seed=42,
augmentation interne desactivee (number_augmentations=1, flip_edges=0,
slide_verts=0), 200 epochs (niter=100, niter_decay=100).

## Dataset final utilise
4241 fichiers valides sur 4899 originaux (86.6%), apres pipeline complet
de nettoyage, reparation topologique (pymeshfix) et decimation a 500 faces
max.

## Meilleur modele (epoch 50)
- Accuracy       : 31.44%
- Precision (macro) : 0.3234
- Recall (macro)    : 0.2918
- F1-score (macro)  : 0.2771
- Nombre de parametres : 1.579 M
- Memoire GPU (pic, inference) : 561.90 Mo
- Temps d'inference moyen : 65.06 ms/echantillon

## Observation cle : overfitting marque
La training loss atteint quasiment 0 des l'epoch ~22, tandis que la test
accuracy plafonne et oscille (24-31%) jusqu'a la fin des 200 epochs, sans
progression nette apres l'epoch 50. Reference importante pour evaluer
l'impact de l'augmentation de donnees.

## Fichiers associes
- resultats/modelnet10_original/test_accuracy_par_epoch.csv
- resultats/modelnet10_original/train_loss_par_batch.csv
- resultats/modelnet10_original/matrice_confusion.csv
- resultats/modelnet10_original/metriques.json
