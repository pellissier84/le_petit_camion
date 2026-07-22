import json
import os

# Script : lister les fichiers audio présents dans un JSON Pyannote
# commande : python lister_fichiers.py
# Ce script lit un fichier JSON contenant une clé "files" et
# extrait tous les noms de fichiers audio associés à cette clé.
# Il affiche la liste dans la console et la sauvegarde dans un
# fichier texte : liste_fichiers_nexsis.txt

def lister_fichiers_audio(nom_fichier_json):
    """
    Lit un fichier JSON Pyannote et extrait la liste des fichiers audio.
    """
    print(f"[INFO] Lecture du fichier {nom_fichier_json}...")

    # Vérification de l'existence du fichier JSON
    if not os.path.exists(nom_fichier_json):
        print(f"[ERREUR] Le fichier '{nom_fichier_json}' est introuvable dans ce dossier.")
        return

    # 1. Chargement sécurisé du JSON
    try:
        # Lecture du JSON en UTF-8 (important pour les caractères accentués)
        with open(nom_fichier_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # Le JSON est mal formé ou contient des erreurs de syntaxe
        print(f"[ERREUR] Le fichier '{nom_fichier_json}' n'est pas un JSON valide.")
        print(f"Détails : {e}")
        return

    # 2. Extraction des noms de fichiers
    liste_fichiers = []

    # Vérification de la structure attendue : présence de la clé "files"
    if isinstance(data, dict) and "files" in data:
        # Parcours de chaque entrée audio dans la liste
        for audio_data in data["files"]:
            # On récupère la valeur associée à la clé "file"
            nom_audio = audio_data.get("file")
            if nom_audio:
                liste_fichiers.append(nom_audio)
    else:
        print("[ERREUR] La structure du JSON n'est pas reconnue (clé 'files' introuvable).")
        return

    # 3. Affichage et sauvegarde

    total = len(liste_fichiers)

    print(f"\n==================================================")
    print(f" {total} fichiers trouvés dans la base Pyannote")
    print(f"==================================================\n")
    
    fichier_sortie = "liste_fichiers_nexsis.txt"
    
    # Écriture dans le fichier texte et affichage console
    with open(fichier_sortie, 'w', encoding='utf-8') as f_out:

        for i, nom in enumerate(liste_fichiers, 1):
            # Affichage console avec numérotation
            ligne = f" {i:3}. {nom}"
            print(ligne)
            # Écriture du nom brut dans le fichier (une ligne par fichier)
            f_out.write(f"{nom}\n")

    print(f"\n[SUCCÈS] La liste brute a été sauvegardée dans '{fichier_sortie}' pour tes copier/coller ! \n")

# Point d'entrée du script
if __name__ == "__main__":
    # Nom du fichier JSON à analyser
    fichier_cible = "segmentation_audios_nettoyes.json" 
    lister_fichiers_audio(fichier_cible)
