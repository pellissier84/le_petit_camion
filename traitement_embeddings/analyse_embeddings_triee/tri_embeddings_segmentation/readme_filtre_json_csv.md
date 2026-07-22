README — Nettoyage des segments audio & filtrage des embeddings

Ce projet contient un script Python permettant de nettoyer des données de segmentation audio et 
de filtrer un fichier CSV d’embeddings afin de ne conserver que les segments valides.

Il est conçu pour fonctionner avec deux fichiers :

    segmentation_audios_Nexsis.json : segmentation audio brute

    embeddings.csv : embeddings associés aux segments

Le script produit deux fichiers nettoyés :

    segmentation_audios_nettoyes.json

    embeddings_nettoyes.csv

Objectifs du script
 
		1. Nettoyage du JSON

		Le script :

			supprime les segments trop courts (durée ≤ 2 secondes)

			reconstruit un identifiant unique pour chaque segment valide :
			nomBase_speaker_compteur

			conserve uniquement les fichiers contenant au moins un segment valide

		2. Filtrage du CSV

		Le script :

			lit chaque ligne du CSV

			extrait l’ID du segment (avant le |)

			conserve uniquement les lignes dont l’ID correspond à un segment valide du JSON
    
Exécution

		Assure-toi que les fichiers suivants sont présents dans le même dossier :

			segmentation_audios_Nexsis.json

			embeddings.csv

		Puis lance :


		python filtre_json_csv.py

		Fichiers générés

			segmentation_audios_nettoyes.json  
			→ JSON filtré, ne contenant que les segments valides

			embeddings_nettoyes.csv  
			→ CSV filtré, ne contenant que les embeddings correspondant aux segments valides
