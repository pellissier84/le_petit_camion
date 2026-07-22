Évaluation Hybride Transcription : WhisperX vs ELAN

Ce projet contient un pipeline d'évaluation NLP (Natural Language Processing)
 conçu pour analyser et quantifier la précision des transcriptions automatiques
 (WhisperX) d'appels d'urgence par rapport à une vérité terrain humaine (annotations ELAN).

Le script evaluate_transcription_EntityRuler.py combine une approche par règles métiers
 (Expressions Régulières via un fichier JSONL) et l'intelligence artificielle statistique
 (NER via spaCy) pour extraire, classer et vérifier la survie des concepts critiques (SDIS 31)
 lors du passage à la machine, tout en calculant le taux d'erreur global (WER).
   
Fonctionnalités Principales

    (nouveau) Calcul du WER Global : Évaluation de la dégradation phonétique complète du texte 
    sans biais lié aux erreurs de diarisation (séparation des locuteurs).

    Extraction Hybride (EntityRuler + NER) : * Captation des concepts métiers stricts
     (Âge, Type de feu, Moyens demandés) via un dictionnaire JSONL externalisé (Nouveau remplace Regex).

     Captation flexible des lieux, personnes et temporalités via le modèle français de spaCy.

    Intégration de Gazetteer Dynamique : Injection automatique et tolérante aux accents des 
    communes de la Haute-Garonne pour une détection géographique robuste.

    Analyse de Rétention d'Information : Rapport détaillé listant les mots critiques
    "sauvés" par l'IA et ceux "perdus", essentiels pour qualifier opérationnellement la fiabilité de Whisper.

    Filtrage Anti-Bruit : Élimination native des faux positifs (ex: "c'", "l'") souvent générés à l'oral.
    
Prérequis et Installation
  Création et activation de l'environnement virtuel
	conda create -n audio_env python=3.10
	conda activate audio_env
  Installation des dépendances Python :
	pip install jiwer
	pip install pympi-ling
	pip install spacy
	
  Téléchargement du modèle linguistique spaCy :
	python -m spacy download fr_core_news_sm
	
Structure des Fichiers Attendus

Pour s'exécuter correctement, le script doit avoir accès aux fichiers suivants dans son répertoire d'exécution :

    evaluate_transcription_EntityRuler.py : Le script principal.

    patterns_sdis.jsonl : Le dictionnaire des règles métiers (Regex/Mots-clés au format spaCy).

    communes-haute-garonne.geojson.json : Le référentiel géographique pour l'extraction des communes.

    Fichiers cibles d'analyse : Votre annotation humaine .eaf (ELAN) et la transcription machine .json (WhisperX).
    
Utilisation

  python evaluate_transcription_EntityRuler.py --eaf "chemin/vers/Appel_annotation.eaf" --json "chemin/vers/transcription_whisper.json"
  
Interprétation de la Sortie (Console)

Le script génère un rapport en trois parties directement dans le terminal :

    Trace d'Analyse : Affiche en temps réel chaque concept métier identifié chez le requérant 
    ou l'opérateur, et vérifie si Whisper l'a correctement transcrit dans la même fenêtre temporelle.

    Bilan Global Comparatif : Affiche le WER Global (Word Error Rate sur le texte intégral) 
    et le Score Hybride par locuteur (pourcentage d'informations métiers conservées).

    Synthèse des Informations (SDIS 31) : Liste récapitulative et classée des informations 
    sécurisées (🟢) et des informations manquantes (🔴), permettant d'identifier rapidement 
    les angles morts du modèle de transcription.
    
    Exemple :
    ===========================================
	BILAN GLOBAL COMPARATIF
	===========================================
	WER Global (Texte complet) : 0.28
	Score Hybride Opérateur    : 28.6%
	Score Hybride Requérant    : 35.7%

	===========================================
	SYNTHÈSE DES INFORMATIONS MÉTIER (SDIS 31)
	===========================================
	🟢 INFORMATIONS SÉCURISÉES PAR L'IA (7) :
		- ACCIDENT_ROUTE                 : 'accident'
		- ACCIDENT_ROUTE                 : 'camion'
		- ACCIDENT_ROUTE                 : 'voiture'
		- ACCIDENT_ROUTE                 : 'voitures'
		- LIEU_INCONNU_IA (LOC)          : 'méditerranée'
		- LOCALISATION_SPECIFIQUE        : 'dans un fossé'

	🔴 INFORMATIONS MANQUANTES (ERREURS WHISPER) (14) :
		- ACCIDENT_ROUTE                 : 'tonneaux'
		- ACCIDENT_ROUTE                 : 'voiture'
		- AGE_VICTIME                    : 'soixante ans'
		- COMMUNES_31                    : 'saint sulpice sur leze'
		- LIEU_INCONNU_IA (LOC)          : 'Lagrace-Dieu'
		- LOCALISATION_SPECIFIQUE        : 'D six cent vingt deux'
		- LOCALISATION_SPECIFIQUE        : 'en contre bas'
		- MESURES                        : '25 mètres'
		- MESURES                        : 'cinquante mètres'
		- MESURES                        : 'quarante mètres'
		- PERSONNE_IA (PER)              : 'Lagrace-Dieu'


    
    
