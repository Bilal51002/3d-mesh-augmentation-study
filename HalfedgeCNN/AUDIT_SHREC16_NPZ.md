# Audit du format de données SHREC16 (.npz)

## Contexte
Le dataset SHREC16 fourni est composé uniquement de fichiers .npz, à convertir
au format .obj attendu par le dataloader HalfedgeCNN (util/util.py,
MESH_EXTENSIONS = ['.obj']).

## Structure du format .npz observée
Chaque fichier .npz contient 4 tableaux :
- `path_index` : shape (253,)
- `vertices` : shape (252, 3)
- `Normal` : shape (252, 3)
- `angles` : shape (252,)

Aucun tableau de faces (connectivité des triangles) n'est présent.

## Tests effectués

### Test 1 — Cohérence sur 10 fichiers différents (T306, T455, T564, T388,
T300, T575, T429, T515, T336, T577)
Résultat : les 10 fichiers ont exactement la même forme :
- vertices.shape = (252, 3) — identique pour tous les fichiers testés
- path_index.shape = (253,)
- path_index min = 0, max = 251
- 252 valeurs uniques dans path_index (sur 253 entrées)

## Analyse

1. **Taille fixe (252 sommets) sur tous les fichiers testés** : incohérent avec
   un mesh brut, dont le nombre de sommets varie naturellement selon l'objet
   (à titre de comparaison, un fichier .off de ModelNet10 analysé en parallèle
   contient 14396 sommets). Ceci indique un pré-traitement de normalisation
   appliqué uniformément à chaque objet du dataset.

2. **path_index couvre exactement les indices 0 à 251 (252 valeurs uniques
   sur 253 entrées)** : structure caractéristique d'une marche qui visite
   chaque sommet du mesh une seule fois (proche d'un chemin hamiltonien),
   avec probablement un retour au sommet de départ (d'où la 253e entrée
   répétant un indice déjà visité).

3. **Absence de tout tableau de faces**, direct ou indirect : path_index est
   une séquence de visite (ordre de parcours), pas une triangulation.

## Conclusion
Ce format .npz correspond à une représentation séquentielle pré-calculée du
mesh (type "random walk on mesh", ex. MeshWalker), destinée à un usage
spécifique différent de HalfedgeCNN. Il ne contient aucune information de
connectivité par faces, condition indispensable pour reconstruire un fichier
.obj valide et exploitable par HalfedgeCNN.

## Statut
Reconstruction d'un .obj impossible à partir des données telles que fournies,
sans risque de fausser la topologie réelle du mesh d'origine.
Question transmise à l'encadrant : existence d'une version brute du dataset
(sommets + faces) pour SHREC16.

## Repo de référence
HalfedgeCNN — https://github.com/IngmarLudwig/HalfedgeCNN
Commit utilisé : a958b67dcd026f6a5cfa027c6f146b3f60ee293d
