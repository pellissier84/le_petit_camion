README — Évaluation métier avancée des transcriptions WhisperX

Objectif du script

Le script evaluate_wer_metier_complet.py permet d’évaluer la qualité des transcriptions WhisperX 
en se concentrant exclusivement sur les concepts métier liés aux appels d’urgence 
(ontologie SDIS, lieux, personnes, mesures, etc.).

Il combine trois sources d’information :

    Ontologie SDIS (Regex) — ton vocabulaire métier spécialisé
    Dictionnaire CISU (EntityRuler spaCy) — patterns issus de ton fichier JSONL
    spaCy NER — détection automatique des entités (LOC, PER, ORG, MISC…)
    

L’objectif est de produire une analyse fine et détaillée des erreurs de WhisperX :

    Correct
    Substitution
    Délétion
    Insertion / hallucination

Ainsi qu’un WER métier (taux d’erreur spécifique aux concepts métier).

Fonctionnement général
	1. Extraction des mots métier

	Le script extrait les concepts métier depuis la référence ELAN via :

		Regex SDIS (ontologie métier)
		EntityRuler CISU (patterns JSONL)
		spaCy NER (rattrapage automatique)

	Chaque concept est catégorisé :
	CONSCIENCE_NEURO, DETRESSE_VITALE, LOCALISATION_SPECIFIQUE, COMMUNES_31, etc.

	2. Analyse des erreurs WhisperX

	Pour chaque concept métier, le script vérifie :

		CORRECT → présent dans WhisperX
		SUBSTITUTION → mot remplacé (avec score de similarité)
		DELETION → mot absent
		INSERTION → hallucinations détectées dans WhisperX

	Les insertions sont validées par spaCy pour distinguer :

		entités nommées plausibles
		mots inventés

	3. Calcul des métriques

	Le script calcule :

		WER métier    
		Précision par catégorie
		Répartition Regex / EntityRuler / spaCy
		Liste complète des concepts détectés

	4. Rapport JSON

	Un fichier rapport_complet_spacy.json est généré pour analyse ultérieure.

Commandes d’exécution

	python evaluate_wer_metier_complet.py --eaf "audio_elan_3540.eaf" --json "audio-3540.json"

Structure du projet
	.
			├── evaluate_wer_metier_complet.py
			├── patterns_sdis_cisu.jsonl        # Dictionnaire métier (EntityRuler)
			├── communes-haute-garonne.geojson  # Gazetteer des communes 31
			├── audio_elan_XXXX.eaf             # Référence ELAN
			├── audio-XXXX.json                 # Transcription WhisperX
			└── rapport_complet_spacy.json      # Rapport généré

Dépendances
	Python

		Python 3.8+
		spaCy (fr_core_news_sm)
		pympi
		difflib
		json / argparse / unicodedata
