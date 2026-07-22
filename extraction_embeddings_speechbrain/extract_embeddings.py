import os
import json
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
# Chemin vers le fichier JSON contenant les horodatages de diarisation
FICHIER_SEGMENTATION = "segmentation_audios_Nexsis.json"
# Dossier source contenant les enregistrements complets en format WAV
DOSSIER_AUDIOS = "data/mes_audios_wav/"          # Dossier où se trouvent tes 3 fichiers .wav
# Dossier cible où seront sauvegardés les tenseurs (.pt) extraits
DOSSIER_SORTIE = "data/embeddings_corpus/" # Dossier où seront rangés les .pt

# Seuil de durée minimum (en secondes). 
# Explication : Les modèles biométriques ont besoin de suffisamment de contexte 
# phonétique pour extraire une signature fiable. Un "oui" ou une respiration 
# de 0.5s génère un vecteur instable qui fausserait les analyses ultérieures.
DUREE_MINIMUM = 1.0 

#  FONCTION  D'EXTRACTION
# ====================================
def generer_embeddings():
    print("Chargement du modèle d'extraction d'embeddings (ECAPA-TDNN)...")
    # Initialisation du modèle SpeechBrain pré-entraîné sur VoxCeleb.
    # Ce modèle transforme un signal audio en un vecteur numérique (souvent de 192 dimensions)
    # représentant les caractéristiques uniques de la voix.
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="tmpdir"
    )
    
    print(f"Ouverture du fichier de segmentation : {FICHIER_SEGMENTATION}")
    with open(FICHIER_SEGMENTATION, 'r', encoding='utf-8') as f:
        donnees = json.load(f)

    # Création du dossier principal de sortie
    Path(DOSSIER_SORTIE).mkdir(parents=True, exist_ok=True)

    # Parcourir chaque fichier audio listé dans le JSON
    for element in donnees['files']:
        nom_fichier_wav = element['file']
        chemin_wav = os.path.join(DOSSIER_AUDIOS, nom_fichier_wav)
        
        # Sécurité : on vérifie que le WAV existe bien sur le disque dur
        if not os.path.exists(chemin_wav):
            print(f"ATTENTION : Le fichier {nom_fichier_wav} est introuvable dans {DOSSIER_AUDIOS}. Ignoré.")
            continue

        # .stem permet de récupérer le nom du fichier sans son extension ".wav"
        # Cela servira d'identifiant unique pour créer le dossier de cet appel
        id_discussion = Path(nom_fichier_wav).stem
        
        # Création d'un sous-dossier dédié à cet appel pour ranger proprement les segments
        dossier_discussion = os.path.join(DOSSIER_SORTIE, id_discussion)
        Path(dossier_discussion).mkdir(exist_ok=True)
        
        print(f"\n--- Traitement de l'audio : {nom_fichier_wav} ---")
        
        # Optimisation RAM : On charge l'intégralité du fichier WAV une seule fois.
        # 'signal' est l'onde acoustique (tenseur), 'fs' (Frequency Sample) est l'échantillonnage (ex: 16000 Hz)
        signal, fs = torchaudio.load(chemin_wav)
        
        segments_valides = 0
        segments_ignores = 0

        # On itère sur chaque prise de parole ('turn') de cet enregistrement
        for idx, turn in enumerate(element['turns']):
            debut = turn['start']
            fin = turn['end']
            locuteur = turn['speaker']
            duree = fin - debut
            
            # 1. Ignorer les segments trop courts
            if duree < DUREE_MINIMUM:
                segments_ignores += 1
                continue
                
            # Conversion du temps (secondes) en "frames" (nombre d'échantillons audio)
            # Ex : Si debut = 2.0s et fs = 16000 Hz, frame_debut = 32000
            frame_debut = int(debut * fs)
            frame_fin = int(fin * fs)
            # Découpage du tenseur audio global pour isoler uniquement ce segment
            segment_audio = signal[:, frame_debut:frame_fin]
            
            # Inférence : Passage dans le réseau de neurones
            # torch.no_grad() désactive le calcul des gradients (utilisé uniquement pour 
            # l'entraînement), ce qui économise énormément de RAM et de temps de calcul.
            with torch.no_grad(): # Pas besoin de gradients, on fait juste de l'inférence
                embedding = classifier.encode_batch(segment_audio)
                
            # Formatage du nom de fichier : 
            # {idx:03d} ajoute des zéros (001, 002) pour garantir un tri chronologique parfait par l'OS
            nom_sortie = f"seg_{idx:03d}_{locuteur}.pt"
            chemin_sortie = os.path.join(dossier_discussion, nom_sortie)
            
            # Sauvegarde de l'empreinte vocale.
            # .squeeze() aplatit le tenseur (ex: de [1, 1, 192] à [192]) pour simplifier 
            # les futurs calculs de distance mathématique.
            torch.save(embedding.squeeze(), chemin_sortie)
            segments_valides += 1

        # Bilan pour le fichier en cours
        print(f"Terminé pour {id_discussion} : {segments_valides} segments générés ({segments_ignores} ignorés car < {DUREE_MINIMUM}s).")

# POINT D'ENTRÉE DU SCRIPT
# ==========================
if __name__ == "__main__":
    generer_embeddings()
    print("\n✅ Extraction terminée avec succès !")
