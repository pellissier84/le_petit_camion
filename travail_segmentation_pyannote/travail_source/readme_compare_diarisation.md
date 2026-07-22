Comparaison visuelle de la diarisation (ELAN vs Pyannote)

Objectif du script

Ce script permet de visualiser et comparer la diarisation issue :

    des annotations manuelles ELAN (.eaf)
    des prédictions automatiques Pyannote (JSON)

L’objectif est de produire un chronogramme double, superposant :

    La vérité terrain (tiers ELAN)
    La diarisation automatique (Pyannote)

Cela permet d’évaluer rapidement :

    la qualité de la segmentation temporelle
    la cohérence des locuteurs
    les erreurs de fusion / fragmentation
    les confusions entre locuteurs

Fonctionnement général

	1. Extraction des segments ELAN

		Chaque tier (piste) est considérée comme un locuteur.
		Le script extrait :

			début (s)
			fin (s)
			nom du locuteur (nom du tier)

	2. Extraction des segments Pyannote

		Le script lit un fichier JSON contenant :

			la liste des fichiers audio analysés
			pour chacun : les segments (turns) avec start, end, speaker

	3. Visualisation

		Le script trace deux chronogrammes :

			Haut : ELAN (vérité terrain)
			Bas : Pyannote (prédictions)

		Chaque locuteur reçoit une couleur unique.
		Les segments sont affichés sous forme de barres horizontales (broken_barh).
		
Exemple d’utilisation

	1. Modifier les chemins dans le script
		CHEMIN_EAF = "audio-1775033540.32214.eaf"
		CHEMIN_JSON = "segmentation_audios_Nexsis.json"
		NOM_DU_WAV_A_CHERCHER = "audio-1775033540.32214.wav"

	2. Lancer le script
	
		python comparaison_diarisation.py
		
	3. Résultat attendu

		Une fenêtre matplotlib s’ouvre avec :

			Graphique 1 : ELAN
			Graphique 2 : Pyannote
			
Structure du projet

			.
			├── comparaison_diarisation.py
			├── segmentation_audios_Nexsis.json
			├── audio-XXXX.eaf
			├── audio-XXXX.wav
			└── README.md

Dépendances
		Python

			pympi-ling
			matplotlib
			json (standard)
