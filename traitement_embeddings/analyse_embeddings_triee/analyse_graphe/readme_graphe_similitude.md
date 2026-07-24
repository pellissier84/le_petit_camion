README — graphe_similitude.py

Objectif du script

Ce projet construit une carte topologique de similitude entre des discussions audio, en se basant sur des embeddings vectoriels et une segmentation pyannote.
Il identifie pour chaque audio les 10 discussions les plus similaires, puis génère :

    un CSV listant les liens de similitude,
    un graphe dirigé (NetworkX + Matplotlib),
    une visualisation mettant en évidence les liens réciproques (noyau dur).

Pipeline complet

	1. Analyse des discussions

		Le script lit un fichier JSON de segmentation pyannote et détecte les discussions où le locuteur cible (ex. operateur) intervient au moins deux fois.
		Cela permet d’éviter les discussions trop courtes ou non pertinentes.

	2. Chargement des embeddings

		Les embeddings sont lus depuis un fichier CSV/TSV au format :		
		utt_id | v1 v2 v3 ... vN
		Ils sont stockés en mémoire dans un dictionnaire Python.
		
	3. Calcul des similitudes

		Pour chaque discussion :

			Extraction des embeddings du locuteur.
			Calcul d’un centroïde global.
			Sélection des top-K segments les plus proches du centroïde.
			Comparaison croisée avec toutes les autres discussions via cosine similarity.

		Seules les 10 meilleures correspondances sont conservées.
		
	4. Construction du graphe

		Noeuds = discussions
		Arêtes = liens Top 10
		Arêtes rouges = liens réciproques (noyau dur)
		Taille des noeuds = nombre de fois où l’audio est ciblé		

	5. Export

		liens_top10_reseau_<timestamp>.csv
		carte_topologique_<timestamp>.png

Organisation des fichiers

		project/
		│--- graphe_similitude.py
		│--- segmentation_audios_nettoyes.json
		│--- data/
		│      |--- embeddings_nettoyes.csv
		│--- resultats_reseau_global/
		│      |--- liens_top10_reseau_*.csv
		│      |--- carte_topologique_*.png

Paramètres importants

    LOCUTEUR_CIBLE : locuteur analysé (ex. "operateur")
    NB_TOP_SEGMENTS : nombre de segments utilisés pour le centroïde robuste
    LIMITE_TOP_CIBLES : nombre de meilleures correspondances conservées
    SOURCE_DONNEES : fichier CSV des embeddings
    FICHIER_JSON_PYANNOTE : segmentation pyannote
    
Exécution

		python graphe_similitude.py
		
Résumé

| Fichier / Dossier                     |            Rôle                           | Obligatoire |
| ------------------------------------- | ----------------------------------------- | ---|
| **graphe_similitude.py**              | Script principal                          | ✔️ |
| **segmentation_audios_nettoyes.json** | Segmentation pyannote (locuteurs + tours) | ✔️ |
| **data/embeddings_nettoyes.csv**      | Embeddings vectoriels                     | ✔️ |
| **resultats_reseau_global/**          | Dossier de sortie (auto‑créé)             | ✔️ |

