README — Analyse de similarité entre segments audio (embeddings)

Ce script permet d’analyser la similarité entre des segments audio appartenant à différentes discussions, 
en utilisant des embeddings (PyTorch, PKL ou CSV).
Il s’appuie sur un fichier JSON issu de Pyannote pour identifier les discussions valides et sur un fichier 
d’embeddings pour effectuer les calculs.

Objectifs du script

	1. Identifier les discussions valides

		Le script lit un JSON Pyannote et sélectionne uniquement les discussions où un locuteur donné (ex. operateur) 
		possède au moins deux segments.
		Cela évite de lancer des calculs sur des discussions inutilisables.
		
	2. Charger les embeddings

		Plusieurs formats sont supportés :

			.pt : fichiers PyTorch individuels
			.pkl : fichiers Kiwano ou PyTorch
			.csv / .tsv : format texte ID_segment | val1 val2 ...

		Le script détecte automatiquement le format et charge les embeddings en conséquence.
		
	3. Calculer les similarités

		Deux types de calculs sont effectués :

			Intra-discussion : similarité entre les segments d’une même discussion
			Inter-discussion : similarité entre une discussion de référence et les autres discussions

		Les similarités sont basées sur la cosine similarity.
		
	4. Générer des rapports

		Le script produit :

			une matrice de similarité (heatmap) avec le nom des discussions segmentées (audio)
			un graphique horizontal comparant les moyennes inter-discussions
			un CSV récapitulatif des scores

		Les résultats sont enregistrés dans un dossier resultats_analyse.

Fichiers attendus

	JSON Pyannote (extrait)


			{
			  "files": [
				{
				  "file": "appel_001.wav",
				  "turns": [
					{ "speaker": "operateur", "start": 0.0, "end": 3.5 },
					{ "speaker": "0", "start": 4.0, "end": 6.0 }
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


python analyse_similitude.py

		Le script te demandera de choisir une discussion de référence.

		Résultats générés

Dans resultats_analyse/ :

    01_matrice_intra_*.png → matrice de similarité intra-discussion
    02_moyennes_horizontales_*.png → comparaison inter-discussions
    rapport_similitudes_*.csv → tableau récapitulatif
    
document de référence complet  
		Méta-données de contexte : Date d'exécution et nom du locuteur cible.

		Volumétrie des calculs : Le nombre exact de paires comparées (pour justifier la robustesse de la moyenne).

		Indicateurs de dispersion : Les scores minimum et maximum, ainsi que l'écart-type, ce qui permet de repérer 
		immédiatement s'il y a des segments très atypiques ou une forte variance dans la voix.  
