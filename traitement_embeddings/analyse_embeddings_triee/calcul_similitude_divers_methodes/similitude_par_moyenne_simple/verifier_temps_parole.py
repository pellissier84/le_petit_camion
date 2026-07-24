import json
"""
Analyse des segments par locuteur (durées + nombre de segments)
Ce script permet d’analyser un fichier JSON issu d’un système de segmentation audio 
(ex. Pyannote) afin d’obtenir, pour chaque discussion :

    le nombre de segments par locuteur
    la liste des durées de ces segments
    un affichage structuré permettant d’identifier les segments courts ou atypiques

Ce script est utile pour :

    vérifier la qualité de la segmentation
    repérer les locuteurs dominants
    détecter les segments trop courts (< 1–2 secondes)
    préparer des statistiques pour un rapport ou une analyse TAL

Fonctionnement

Le script :

    Charge un fichier JSON contenant des discussions audio.
    Pour chaque discussion :

        extrait les segments (turns)
        calcule la durée de chaque segment (end - start)
        regroupe les durées par locuteur
"""

def analyser_segments_locuteurs(chemin_json):
    """
    Analyse un fichier JSON contenant des segments audio annotés.
    Pour chaque discussion, affiche :
        - le locuteur
        - le nombre de segments
        - la liste des durées de chaque segment

    Paramètres
    ----------
    chemin_json : str
        Chemin vers le fichier JSON Pyannote nettoyé.

    Fonctionnement
    --------------
    - Charge le JSON.
    - Parcourt chaque discussion (fichier audio).
    - Pour chaque segment :
        * calcule la durée (end - start)
        * ajoute la durée dans une liste associée au locuteur
    - Affiche un tableau structuré dans le terminal.
    """

    # Ouverture et lecture du fichier JSON
    with open(chemin_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Entête du tableau affiché
    print(f"{'Discussion':<30} | {'Locuteur':<12} | {'Nb Segments':<12} | {'Durées des segments (s)'}")
    print("-" * 100)

    # Parcours de chaque discussion
    for file_data in data.get('files', []):

        # Nom de la discussion sans extension (.wav)
        nom_base = file_data['file'].replace('.wav', '')
        # Dictionnaire pour stocker les listes de durées par locuteur
        details_locuteurs = {}

        # Parcours des segments ("turns")
        for turn in file_data.get('turns', []):
            # Durée du segment arrondie à 2 décimales
            duree = round(turn['end'] - turn['start'], 2)
            # Locuteur associé au segment
            speaker = turn['speaker']
            
            # Initialisation de la liste si nécessaire
            if speaker not in details_locuteurs:
                details_locuteurs[speaker] = []
            # Ajout de la durée dans la liste du locuteur
            details_locuteurs[speaker].append(duree)

        # Affichage des résultats pour cette discussion
        for speaker, durees in details_locuteurs.items():
            nb_segments = len(durees)
            # On affiche la liste des durées pour voir les segments courts
            liste_durees = ", ".join([str(d) for d in durees])
            print(f"{nom_base:<30} | {speaker:<12} | {nb_segments:<12} | {liste_durees}")
        
        print("-" * 100)

# Point d'entrée du script
if __name__ == "__main__":
    analyser_segments_locuteurs("segmentation_audios_nettoyes.json")
