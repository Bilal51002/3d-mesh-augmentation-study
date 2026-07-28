# Conversion des données — ModelNet10

## Format d'origine
.off (Object File Format) — structure standard : en-tête avec nombre de
sommets/faces, puis coordonnées des sommets, puis indices des faces.

## Format cible
.obj — requis par le dataloader HalfedgeCNN (util/util.py, MESH_EXTENSIONS = ['.obj'])

## Méthode
Conversion via la librairie trimesh (https://github.com/mikedh/trimesh),
sans post-traitement (process=False) pour préserver fidèlement la géométrie
d'origine sans fusion de sommets dupliqués ni modification de la topologie.

## Résultat
4899 fichiers .off convertis en 4899 fichiers .obj (aucune perte, 0 erreur).
Répartition par classe : bathtub (156), bed (615), chair (989), desk (286),
dresser (286), monitor (565), night_stand (286), sofa (780), table (492),
toilet (444).
Structure de dossiers d'origine (classe/train|test/) conservée à l'identique.
Dataset converti sauvegardé sur Kaggle Datasets : modelnet10-obj-halfedgecnn

