# Simplification des meshes — ModelNet10

## Problème identifié
Analyse de la distribution du nombre de faces sur les 4899 fichiers .obj
convertis (voir CONVERSION_DONNEES.md) :
- Minimum : 60 faces
- Maximum : 403575 faces
- Moyenne : 10319 faces
- Médiane : 2930 faces
- P90 : 28119 / P95 : 43466 / P99 : 103942

Ces valeurs sont largement incompatibles avec les résolutions utilisées par
HalfedgeCNN sur ses datasets de référence (ex. shrec_16_settings.txt utilise
--number_input_faces 500). Un nombre de faces aussi élevé rendrait
l'entraînement extrêmement lent et gourmand en mémoire GPU.

Cette pratique de simplification est cohérente avec celle documentée par
l'architecture MeshNet (iMoonLab/MeshNet), qui recommande explicitement de
simplifier les meshes ModelNet à un nombre de faces maximal avant tout usage.

## Méthode de simplification

1. Nettoyage préalable de chaque mesh (trimesh 4.12.2) :
   - merge_vertices() : fusion des sommets dupliqués
   - update_faces(nondegenerate_faces()) : suppression des faces d'aire nulle
   - update_faces(unique_faces()) : suppression des faces dupliquées
   - remove_unreferenced_vertices()

2. Décimation quadrique itérative (simplify_quadric_decimation), par paliers
   successifs (réduction ~30-50% par itération, jusqu'à 15 itérations max),
   ciblant initialement 1500 faces.

## Résultat de la simplification
- 4899 fichiers traités
- Cible initiale (1500 faces) atteinte pour l'immense majorité des fichiers
- 320 fichiers résistants après nettoyage + décimation itérative, dont
  la majorité proche de la cible (231/320 sous 5000 faces)

## Décision finale
Plutôt que de forcer une décimation extrême sur des cas structurellement
résistants (probablement des meshes non-manifold avec de nombreux composants
déconnectés), le seuil retenu pour --number_input_faces est fixé à 5000
(au lieu de 1500), ce qui couvre 4810/4899 fichiers (98.2%) sans dégradation
supplémentaire de leur géométrie.

## Fichiers exclus
89 fichiers (1.8% du dataset) dépassant 5000 faces après simplification ont
été déplacés hors du dataset d'entraînement, vers un dossier séparé
(obj_exclus_trop_complexes), plutôt que supprimés, pour traçabilité.

Répartition des exclusions par classe (cas les plus représentés) : sofa,
chair, night_stand, bed — classes contenant des objets aux géométries les
plus complexes (courbes, composants multiples).

## Hyperparamètre retenu
--number_input_faces 5000 (voir modelnet10_settings.txt)

## Limite reconnue
Ce choix introduit un biais potentiel : les classes avec les géométries les
plus complexes (sofa, chair notamment) perdent proportionnellement plus
d'échantillons. Ce point sera mentionné dans les limites de l'étude.
