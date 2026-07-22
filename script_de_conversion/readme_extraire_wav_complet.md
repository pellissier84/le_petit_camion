Extraction et Concaténation Audio

Ce script Python permet d'automatiser le découpage et la concaténation de fichiers 
audio basés sur une segmentation JSON (issue de pyannote). 
Il est conçu pour faciliter le traitement des enregistrements d'appels des services d'urgence (SDIS).

Fonctionnalités

    Sélection interactive : Liste automatiquement les fichiers .wav disponibles dans votre dossier de travail.

    Mappage personnalisé : Renomme automatiquement les locuteurs (ex: "operateur" devient "OP-SDIS1", "0" devient "requerant0", etc.).

    Extraction segmentée : Découpe les enregistrements originaux en petits segments par locuteur.

    Concaténation intelligente : Fusionne tous les segments d'un même locuteur en un seul fichier audio complet.

    Organisation propre : Sépare les fichiers temporaires des résultats finaux dans des dossiers dédiés.

Prérequis

    Python 3.x

    FFmpeg : Doit être installé et accessible dans votre PATH système (requis pour le traitement audio).

    Bibliothèques Python :
    Bash

    pip install pydub

Structure du projet
Plaintext

.
├── extraire_wav_complet.py            # Le script principal
├── segmentation_audios_Nexsis.json # Fichier de segmentation (Pyannote)
├── mes_audios_wav/            # Placez vos fichiers sources ici
├── segments_extraits/         # Dossier généré (segments individuels)
└── segments_complets_extraits/# Dossier généré (résultats finaux)

Utilisation

    Placez vos fichiers .wav dans le dossier mes_audios_wav/.

    Assurez-vous que votre fichier de segmentation segmentation_audios_Nexsis.json est à la racine.

    Lancez le script :
    Bash

    python extraire_wav_complet.py 

    Suivez les instructions à l'écran : choisissez le numéro du fichier audio que vous souhaitez traiter.

Configuration

Vous pouvez modifier le dictionnaire MAPPING_LOCUTEURS au début du script 
pour ajuster les noms des locuteurs selon vos besoins spécifiques :
Python

MAPPING_LOCUTEURS = {
    "operateur": "OP-SDIS1",
    "0": "requerant0",
    # Ajoutez vos nouveaux locuteurs ici
}
