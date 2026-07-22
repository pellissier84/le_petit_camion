import json
import os
import glob
import subprocess
import sys
"""
Extraction automatique des intervenants depuis un fichier json (pyannote)
# Les segments audio correspondants sont extraits du fichier source et concaténés en un seul fichier par tier/valeur.
# Les fichiers résultants sont nommés de manière claire
- Menu interactif pour le choix des fichiers wav
- SAUVEGARDE des petits fichiers .wav (segments) avec le numéro de l'appel (segments_extraits)
- CONCATÉNATION en un gros fichier .wav par locuteur (segments_complets_extraits)
# audio-1775031826.41778.wav
# audio-1775031968.41843.wav
# audio-1775033540.32214.wav
- recuperartion du nom du fichier + locuteur +  numero d'ordre.wav (ajout de complet pour la concatenation
"""
# ==========================================
# CONFIGURATION ET MAPPING
# ==========================================
MAPPING_LOCUTEURS = {
    "operateur": "OP-SDIS1",
    "0": "requerant0",
    "1": "requerant1",
    "2": "requerant2",
    "3": "requerant3",
    "4": "requerant4"
}

DOSSIER_WAV = "mes_audio_wav"
FICHIER_JSON = "segmentation_audios_Nexsis.json"
DOSSIER_SEGMENTS = "segments_extraits"
DOSSIER_COMPLETS = "segments_complets_extraits" # Nouveau dossier

def extraire_segment_ffmpeg(audio_source, debut, fin, fichier_sortie):
    duree = fin - debut
    cmd = [
        'ffmpeg', '-i', audio_source, '-ss', str(debut), '-t', str(duree),
        '-acodec', 'pcm_s16le', '-ar', '16000', '-y', fichier_sortie
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True

def concatener_segments(segments, fichier_sortie):
    if not segments: return False
    with open('temp_concat_list.txt', 'w') as f:
        for seg in segments: f.write(f"file '{seg}'\n")
            
    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'temp_concat_list.txt', '-c', 'copy', '-y', fichier_sortie]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists('temp_concat_list.txt'): os.remove('temp_concat_list.txt')
    return True

def main():
    # 1. Sélection
    fichiers_disponibles = glob.glob(os.path.join(DOSSIER_WAV, "*.wav"))
    if not fichiers_disponibles:
        print(f"Aucun fichier .wav dans '{DOSSIER_WAV}'.")
        return

    print("Fichiers disponibles :")
    for i, f in enumerate(fichiers_disponibles, 1):
        print(f"{i} : {os.path.basename(f)}")
    
    try:
        choix = int(input("\nNuméro du fichier à traiter : ")) - 1
        audio_source = fichiers_disponibles[choix]
    except (ValueError, IndexError):
        print("Choix invalide.")
        return

    # 2. Préparation des dossiers
    nom_base = os.path.splitext(os.path.basename(audio_source))[0]
    os.makedirs(DOSSIER_SEGMENTS, exist_ok=True)
    os.makedirs(DOSSIER_COMPLETS, exist_ok=True) # Création du dossier cible
    
    with open(FICHIER_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 3. Traitement
    segments_par_locuteur = {}
    for fichier in data.get('files', []):
        if fichier['file'] == os.path.basename(audio_source):
            for turn in fichier.get('turns', []):
                locuteur_brut = str(turn['speaker'])
                locuteur_final = MAPPING_LOCUTEURS.get(locuteur_brut, f"locuteur_{locuteur_brut}")
                
                if locuteur_final not in segments_par_locuteur:
                    segments_par_locuteur[locuteur_final] = []
                segments_par_locuteur[locuteur_final].append(turn)

    for locuteur, segments in segments_par_locuteur.items():
        fichiers_segments = []
        print(f"\n🎤 Traitement de {locuteur}...")
        
        for i, seg in enumerate(segments, 1):
            nom_seg = f"{DOSSIER_SEGMENTS}/{nom_base}_{locuteur}_{i:03d}.wav"
            if extraire_segment_ffmpeg(audio_source, seg['start'], seg['end'], nom_seg):
                fichiers_segments.append(nom_seg)
        
        # Concaténation dans le nouveau dossier
        nom_fichier_concat = os.path.join(DOSSIER_COMPLETS, f"{nom_base}_{locuteur}_complet.wav")
        if concatener_segments(fichiers_segments, nom_fichier_concat):
            print(f"  ✅ Généré : {nom_fichier_concat}")

    print(f"\nOpération terminée. Les fichiers complets sont dans '{DOSSIER_COMPLETS}/'.")

if __name__ == "__main__":
    main()
