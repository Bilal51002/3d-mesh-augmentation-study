"""
Script de conversion .off vers .obj pour ModelNet10
Utilisé pour préparer les données au format attendu par HalfedgeCNN
(voir util/util.py : MESH_EXTENSIONS = [\'.obj\'])
"""

import trimesh
from pathlib import Path


def convertir_off_vers_obj(dossier_source, dossier_destination):
    """
    Parcourt dossier_source (structure classe/train|test/*.off),
    convertit chaque .off en .obj, et reproduit la même structure
    dans dossier_destination.
    """
    nb_convertis = 0
    nb_erreurs = 0
    erreurs_details = []

    for chemin_off in Path(dossier_source).rglob(\'*.off\'):
        chemin_relatif = chemin_off.relative_to(dossier_source)
        chemin_obj = Path(dossier_destination) / chemin_relatif.with_suffix(\'.obj\')
        chemin_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            mesh = trimesh.load(chemin_off, file_type=\'off\', process=False)
            mesh.export(chemin_obj, file_type=\'obj\')
            nb_convertis += 1
            if nb_convertis % 200 == 0:
                print(f"{nb_convertis} fichiers convertis...")
        except Exception as e:
            nb_erreurs += 1
            erreurs_details.append((str(chemin_off), str(e)))

    print(f"\nTerminé : {nb_convertis} convertis, {nb_erreurs} erreurs")
    if erreurs_details:
        print("Détail des erreurs (max 10 affichées) :")
        for chemin, err in erreurs_details[:10]:
            print(f"  {chemin} : {err}")

    return nb_convertis, nb_erreurs


if __name__ == "__main__":
    dossier_source = "CHEMIN_VERS_MODELNET10_OFF"
    dossier_destination = "CHEMIN_VERS_MODELNET10_OBJ"
    convertir_off_vers_obj(dossier_source, dossier_destination)
