# Entraînement test — HalfedgeCNN sur ModelNet10

## Objectif
Valider que le pipeline complet (chargement des données, calcul mean/std,
construction du modèle, entraînement, test, sauvegarde) fonctionne de bout
en bout avant de lancer l'entraînement complet.

## Configuration finale retenue

--number_input_faces 500
--pool_res 1200 900 600 360
--seed 42
--number_augmentations 1 --flip_edges 0 --slide_verts 0 (augmentation interne
désactivée, voir AUDIT_AUGMENTATION.md)

Ces valeurs reprennent exactement celles utilisées officiellement par les
auteurs de HalfedgeCNN sur SHREC16 (voir scripts/settings/shrec_16_settings.txt
du repo officiel), après plusieurs tentatives avec des valeurs plus élevées
(number_input_faces=5000) qui se sont révélées instables (voir section
"Historique des tentatives" ci-dessous).

## Résultat du test (1 epoch)
- Training Loss : 2.246
- Test Accuracy : 10.8% (proche du hasard, 10 classes -> ~10% attendu avec
  1 seule epoch, cohérent et attendu)
- Temps total : 316 secondes (~5 min) pour 1 epoch sur GPU Tesla P100
- Batchs entraînement : 863 OK / 3 ignorés (0.35%)
- Batchs test : 194 OK / 1 ignoré (0.5%)

## Conclusion
Le pipeline est validé techniquement. L'accuracy faible est normale et
attendue pour un test à 1 epoch ; elle n'est pas représentative de la
performance finale du modèle. Prêt pour l'entraînement complet (étape 8).

## Historique des tentatives (résumé)
1ère tentative : number_input_faces=5000, pool_res=4000/3000/2000/1200
(valeurs inventées par extrapolation, non testées par les auteurs) : échecs
répétés (faces d'aire nulle, topologie non-manifold, crash de l'algorithme
de pooling par manque de candidats valides dans le tas de fusion).

2ème tentative : reprise de la configuration officielle éprouvée sur
SHREC16 (500 faces, pool_res 1200/900/600/360) : fonctionne, avec un taux
résiduel de batchs problématiques (<1%) géré via train_robuste.py.

## Limite reconnue
Le choix de number_input_faces=500 (au lieu de 5000) impose une
simplification plus importante des meshes ModelNet10, qui contiennent
nativement beaucoup plus de détails géométriques que SHREC16. Ce choix
est un compromis pragmatique entre fidélité géométrique et stabilité de
l'algorithme de pooling propre à cette architecture. Ce point sera discuté
dans les limites de l'étude comparative.
