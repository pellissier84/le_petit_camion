README — Analyse de similarité entre profils vocaux (agrégation + split‑half)

Ce script permet d’analyser la similarité entre des discussions audio en utilisant des embeddings vocaux (PyTorch, PKL ou CSV).
Il s’appuie sur un JSON Pyannote pour identifier les discussions valides, puis calcule :

    la similarité intra-discussion (avec agrégation : centroïde ou pondération par durée)

    la similarité inter-discussion (profil vocal vs profil vocal)

    des visualisations (heatmap + barres horizontales)

    un rapport CSV récapitulatif

Ce script est une version avancée d'une analyse précédente :
	il introduit la méthode d’agrégation, la pondération par durée, et la méthode split‑half pour stabiliser les profils vocaux.
	
Fonctionnalités principales

		1. Analyse du JSON Pyannote

			Vérifie quelles discussions contiennent au moins deux segments du locuteur cible.
			Extrait les durées des segments pour la pondération.

		2. Chargement des embeddings

		Formats supportés :

			.pt → fichiers PyTorch individuels
			.pkl → embeddings Kiwano ou PyTorch
			.csv / .tsv → format texte ID | val1 val2 ...

		3. Agrégation des segments

			Deux méthodes disponibles :
			Méthode	Description
			centroide	moyenne simple des vecteurs
			ponderation	moyenne pondérée par la durée des segments

		4. Calculs de similarité

			Intra-discussion :

				comparaison segment → profil global
				split‑half (pairs vs impairs)

			Inter-discussion :

				comparaison profil vocal référence → profil vocal cible

		5. Visualisations

			Heatmap intra-discussion
			Barres horizontales inter-discussion
			Export CSV
    
Fichiers attendus
	JSON Pyannote


			{
			  "files": [
				{
				  "file": "appel_001.wav",
				  "turns": [
					{ "speaker": "operateur", "start": 0.0, "end": 3.5 },
					{ "speaker": "operateur", "start": 4.0, "end": 7.0 }
				  ]
				}
			  ]
			}

	Embeddings CSV

			appel_001_operateur_1 | 0.12 0.55 -0.33 ...
			appel_001_operateur_2 | 0.14 0.58 -0.31 ...

Exécution

	Assure-toi que les fichiers suivants existent :

		segmentation_audios_nettoyes.json
		embeddings_nettoyes.csv

Puis lance :


	python tester_methode1.py

	Le script te demandera de choisir une discussion de référence.
	
Résultats générés

	Dans resultats_analyse/ :

			01_matrice_intra_*.png → matrice de similarité
			02_moyennes_horizontales_*.png → comparaison inter-discussions
			rapport_similitudes_*.csv → tableau récapitulatif

