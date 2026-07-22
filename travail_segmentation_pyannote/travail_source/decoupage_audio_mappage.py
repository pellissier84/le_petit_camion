import json
import os
import glob
from pydub import AudioSegment

# respect des noms originels des fichiers wav
# extractions des segments wav d'un locuteur
# sans concatenation de wav
# meme fonctionnement que decoupage_audio_liste.py sans le mapping des locuteurs

# ==========================================
# CONFIGURATION
# ==========================================
MAPPING_LOCUTEURS = {
    "operateur": "OP-SDIS1",
    "0": "requerant0",
    "1": "requerant1",
    "2": "requerant2",
    "3": "requerant3",
    "4": "requerant4"
}

DOSSIER_WAV = "mes_audios_wav"
FICHIER_JSON = "segmentation_audios_Nexsis.json" 
DOSSIER_SORTIE = "locuteurs_extraits"

def extraire_wav_par_locuteur(chemin_json, chemin_wav, nom_etiquette_json, dossier_sortie):
    """Découpe un fichier audio en fichiers distincts par locuteur avec renommage."""
    
    # Lecture du JSON
    try:
        with open(chemin_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON '{chemin_json}' est introuvable.")
        return

    segments = None
    for fichier in data.get('files', []):
        if fichier['file'] == nom_etiquette_json:
            segments = fichier.get('turns', [])
            break
            
    if segments is None:
        print(f"Erreur : Aucun segment trouvé pour '{nom_etiquette_json}' dans le JSON.")
        return

    # Chargement audio
    audio_complet = AudioSegment.from_file(chemin_wav)
    os.makedirs(dossier_sortie, exist_ok=True)
    
    compteurs = {}
    nom_base = os.path.splitext(os.path.basename(chemin_wav))[0]

    for seg in segments:
        locuteur_brut = str(seg['speaker'])
        locuteur_final = MAPPING_LOCUTEURS.get(locuteur_brut, f"locuteur_{locuteur_brut}")
        
        debut_ms = int(seg['start'] * 1000)
        fin_ms = int(seg['end'] * 1000)
        
        morceau = audio_complet[debut_ms:fin_ms]
        compteurs[locuteur_final] = compteurs.get(locuteur_final, 0) + 1
        
        nom_fichier_sortie = f"{nom_base}_{locuteur_final}_{compteurs[locuteur_final]}.wav"
        morceau.export(os.path.join(dossier_sortie, nom_fichier_sortie), format="wav")
        print(f"  -> Exporté : {nom_fichier_sortie}")

# ==========================================
# SÉLECTION ET EXÉCUTION
# ==========================================
def main():
    # Lister les fichiers WAV
    fichiers_disponibles = glob.glob(os.path.join(DOSSIER_WAV, "*.wav"))
    
    if not fichiers_disponibles:
        print(f"Aucun fichier .wav trouvé dans le dossier '{DOSSIER_WAV}'.")
        return

    print("Fichiers audio disponibles :")
    for i, f in enumerate(fichiers_disponibles):
        print(f"{i+1} : {os.path.basename(f)}")

    # Choix utilisateur
    try:
        choix = int(input("\nEntrez le numéro du fichier à découper : ")) - 1
        if 0 <= choix < len(fichiers_disponibles):
            fichier_choisi = fichiers_disponibles[choix]
            nom_fichier = os.path.basename(fichier_choisi)
            
            print(f"\nTraitement de : {nom_fichier}...")
            extraire_wav_par_locuteur(FICHIER_JSON, fichier_choisi, nom_fichier, DOSSIER_SORTIE)
        else:
            print("Numéro invalide.")
    except ValueError:
        print("Veuillez entrer un nombre valide.")

if __name__ == "__main__":
    main()
