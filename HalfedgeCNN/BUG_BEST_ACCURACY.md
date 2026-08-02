# Bug corrige — best_accuracy reinitialisee a chaque reprise

## Probleme
Dans train_robuste.py (et train.py original), la variable best_accuracy est
initialisee a 0.0 a chaque lancement du script. Lors d'une reprise via
--continue_train (nouveau processus Python, nouvelle session Kaggle), cette
reinitialisation fait que le premier test apres reprise est automatiquement
sauvegarde comme "best", meme si son accuracy est inferieure au veritable
meilleur score obtenu avant l'interruption.

## Impact constate
Sur l'entrainement ModelNet10 original (200 epochs, 3 sessions) :
le vrai meilleur score (31.4%, epoch 50) a ete ecrase par un score inferieur
(28.9%, epoch 194) lors d'une reprise ulterieure. Corrige manuellement en
recuperant l'ancien best_net.pth depuis une sauvegarde intermediaire.

## Correction appliquee
best_accuracy est desormais persistee dans un fichier
checkpoints/<name>/best_accuracy.json a chaque nouveau record, et relue au
demarrage si --continue_train est actif.

## Verification
Accuracy recalculee avec le bon checkpoint (epoch 50) : 31.44%,
coherente avec la valeur observee pendant l'entrainement (31.4%).
