import json
import os
from pydub import AudioSegment

# Script de découpage audio par locuteur
# ---------------------------------------
# Ce script lit un fichier JSON issu d'une diarisation (ex : pyannote)
# et découpe automatiquement un fichier WAV en segments individuels
# pour chaque locuteur, avec une numérotation séquentielle.
# decoupage des wav de la liste json issu de pyannote
# decoupage en locuteur et par intervention et numero d'ordre
# Format du nom : audio-XXXXX_locuteur_ordre.wav
# commande : python decoupage_audio_liste.py
# sortie dans un fichier locuteurs_extraits
# requis : python 3.7 min, pydub , ffmpeg

def extraire_wav_par_locuteur(chemin_json, chemin_wav, nom_etiquette_json, dossier_sortie="locuteurs_extraits"):
    """
    Découpe un fichier audio en fichiers distincts par locuteur avec une numérotation séquentielle.
    """
    # Création du dossier de sortie si nécessaire
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    # 1. Lecture du JSON
    try:
        with open(chemin_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {chemin_json} est introuvable.")
        return

    # Recherche des segments correspondant au fichier audio
    segments = None
    for fichier in data.get('files', []):
        # On cherche l'entrée JSON dont le nom correspond au WAV à traiter
        if fichier['file'] == nom_etiquette_json:
            segments = fichier.get('turns', [])
            break
            
    # Si aucun segment n'est trouvé, on passe au fichier suivant
    if not segments:
        print(f"  ! Aucun segment trouvé pour {nom_etiquette_json}, passage au suivant.")
        return

    # 2. Chargement de l'audio
    try:
        audio_complet = AudioSegment.from_file(chemin_wav)
    except Exception as e:
        print(f"  ! Impossible de lire {chemin_wav}. Erreur : {e}")
        return

    # 3. Découpage et sauvegarde

    # Compteur par locuteur pour numéroter les segments
    compteurs = {}
    nom_base = os.path.splitext(os.path.basename(chemin_wav))[0]

    for seg in segments:
        locuteur = seg['speaker']
        # Conversion secondes → millisecondes
        debut_ms = int(seg['start'] * 1000)
        fin_ms = int(seg['end'] * 1000)
        
        # Extraction du segment audio
        morceau = audio_complet[debut_ms:fin_ms]
        
        # Incrémentation du compteur pour ce locuteur
        compteurs[locuteur] = compteurs.get(locuteur, 0) + 1
        num_ordre = compteurs[locuteur]
        
        # Construction du nom du fichier de sortie
        # Format du nom : audio-XXXXX_locuteur_ordre.wav
        nom_fichier_sortie = f"{nom_base}_{locuteur}_{num_ordre}.wav"
        chemin_sortie = os.path.join(dossier_sortie, nom_fichier_sortie)
        
	# Export du segment
        morceau.export(chemin_sortie, format="wav")
    
    print(f"  ✓ Traitement terminé pour {nom_etiquette_json}")

# ==========================================
# PARAMÈTRES PRINCIPAUX
# ==========================================

FICHIER_JSON = "segmentation_audios_Nexsis.json"
LISTE_FICHIERS = "liste_fichiers_nexsis.txt"

# Exécution du traitement sur la liste de fichiers WAV
if os.path.exists(LISTE_FICHIERS):
    with open(LISTE_FICHIERS, 'r') as f:
        # On récupère chaque nom de fichier WAV, une ligne par fichier
        fichiers_a_traiter = [ligne.strip() for ligne in f if ligne.strip()]

    print(f"Début du traitement de {len(fichiers_a_traiter)} fichiers...")
    
    for nom_wav in fichiers_a_traiter:
        print(f"Traitement de : {nom_wav}")
        # On suppose que les fichiers wav sont dans le dossier courant ou un dossier défini
        extraire_wav_par_locuteur(FICHIER_JSON, nom_wav, nom_wav)
        
    print("\nTous les fichiers ont été traités !")
else:
    print(f"Erreur : Le fichier liste {LISTE_FICHIERS} est introuvable.")
