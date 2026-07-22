README — Évaluation du Diarization Error Rate (DER) entre ELAN et Pyannote

Objectif du script

		Ce script calcule le DER (Diarization Error Rate) entre :

			la vérité terrain issue des annotations manuelles ELAN (.eaf)
			la diarisation automatique produite par Pyannote (JSON Nexsis)

		Il permet d’obtenir un rapport métrique détaillé comprenant :

			DER global
			Taux d’omissions
			Taux de fausses alertes
			Taux de confusions

		Le tout sur une zone temporelle contrôlée (ex. : les 4 premières minutes).
		
Fonctionnement général
	1. Chargement de la vérité terrain (ELAN)

		Le script :

			charge le fichier .eaf
			convertit chaque annotation en segment temporel
			ignore certains tiers (ex. : Ac_ev = bruit / événements non verbaux)
			construit une annotation Pyannote compatible

	2. Chargement des prédictions Pyannote
	
		Depuis le fichier JSON Nexsis :

			recherche du fichier audio correspondant
			extraction des segments turns
			construction d’une annotation Pyannote

	3. Calcul du DER

		Le script utilise :
		pyannote.metrics.diarization.DiarizationErrorRate

		Avec :

			UEM (Un-partitioned Evaluation Map) = zone d’évaluation (ex. 0 → 240 s)
			comparaison segment par segment
			extraction des composantes du DER

	4. Rapport final

		Le script affiche :

			DER global
			% d’omissions
			% de fausses alertes
			% de confusions
			
Exemple d’utilisation

1. Modifier les paramètres

		CHEMIN_EAF = "audio-1775033540.32214.eaf"
		CHEMIN_JSON = "segmentation_audios_Nexsis.json"
		NOM_ETIQUETTE_DANS_JSON = "audio-1775033540.32214.wav"

		TIERS_A_IGNORER = ["Ac_ev"]
		TEMPS_LIMITE_SECONDES = 240.0

2. Lancer le script

		python comparaisin_metrique_diarisation_V1.py

3. Exemple de sortie

	RAPPORT D'ÉVALUATION CORRIGÉ (DER)
	
		Évalué uniquement sur les 240 premières secondes.
		==================================================
		DER Global          : 18.42 %
		--------------------------------------------------
		  - Omissions       : 7.91 %
		  - Fausses Alertes : 5.12 %
		  - Confusions      : 5.39 %
		==================================================

Structure du projet

		.
		├── evaluation_der.py
		├── segmentation_audios_Nexsis.json
		├── audio-XXXX.eaf
		├── audio-XXXX.wav
		└── README.md

Dépendances
		Python

			pyannote.core
			pyannote.metrics
			pympi-ling
			json (standard)


