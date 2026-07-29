# Configuration des hyperparamètres — ModelNet10

## number_input_faces = 5000
Déterminé après analyse de la distribution du nombre de faces des meshes
simplifiés (voir SIMPLIFICATION_MESHES.md). Couvre 4810/4899 fichiers (98.2%)
du dataset converti et nettoyé.

## pool_res = 4000 3000 2000 1200
Résolutions de pooling successives, définies par analogie avec le ratio de
réduction utilisé dans shrec_16_settings.txt (officiel), qui applique une
réduction progressive d'environ 25% par étape à partir de
number_input_faces=500 (1200 900 600 360), soit une réduction totale
d'environ 70%.
Le même ratio de réduction est appliqué ici à partir de
number_input_faces=5000, donnant : 4000 (-20%) / 3000 (-25%) / 2000 (-33%) /
1200 (-40%), soit une réduction totale similaire (-76%).
Ces valeurs sont indicatives et seront ajustées si besoin après le premier
entraînement test (étape 7), en observant la stabilité de l'entraînement
et l'usage mémoire GPU.

## seed = 42
Valeur arbitraire fixée pour la reproductibilité. Rappel (voir
options/base_options.py du repo officiel) : la reproductibilité exacte
n'est garantie que sur CPU (--gpu_ids -1) ; sur GPU, certaines opérations
CUDA restent non-déterministes malgré le seed fixé.

## gpu_ids = 0
Utilisation du premier GPU disponible (Tesla P100 sur Kaggle).

## Note
Ce fichier de settings n'existe pas nativement dans le repo officiel
HalfedgeCNN (seuls shrec_16, human_seg, coseg_*, cubes sont fournis).
Sa création constitue une extension documentée du repo pour couvrir le
dataset ModelNet10, conformément à la section 5.1 du cahier des charges.
