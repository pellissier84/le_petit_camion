README du script similitude_trois_meilleurs_cluster.py

Premiere etape du traitement des similitudes inter discussions par rapport au resultat de l'intra discussion.
On recupere les discussions (appeler cluster) dont la similitude apres comparaison est superieure au resultat de l'intra.
Rq : dans un deuxieme temps, on traitera avec un autre script un analyse croisee (analyse_croisée_json_cluster.py)


A. Ce que fait ce script (explication claire et complète)

	Ce script réalise une analyse automatique de similitude entre discussions audio, en utilisant 
	des embeddings et une segmentation pyannote.
	Il fonctionne en boucle, en prenant chaque discussion comme référence, puis en comparant cette référence à toutes les autres.

Voici le pipeline exact :

	Étape 1 — Analyse du JSON pyannote

			Le script :

				lit segmentation_audios_nettoyes.json,
				identifie les discussions où le locuteur cible (ex. operateur) parle au moins deux fois,
				trie les discussions par nombre de segments.

			Objectif : ne garder que les discussions avec assez de matière pour une analyse robuste.
			
	Étape 2 — Chargement des embeddings

			Le script peut charger :

				soit un CSV/TSV contenant les embeddings,
				soit un PKL Kiwano,
				soit des fichiers .pt individuels.

			Il extrait ensuite les embeddings correspondant à :

				la discussion,
				le locuteur cible.

	Étape 3 — Calculs mathématiques

			Pour chaque discussion :
			A. Intra-discussion

				Calcul du centroïde global de la discussion.
				Sélection des Top‑K segments les plus proches du centroïde.
				Calcul de la similarité entre ce profil robuste et les autres segments de la discussion.

				Production :

					scores individuels,
					moyenne intra-discussion,
					écart-type.

			B. Inter-discussion

			Pour chaque autre discussion :

				Calcul du profil robuste Top‑K.
				Calcul de la similarité cosinus entre les deux profils.
				Stockage des résultats.

	Étape 4 — Génération des rapports

			Pour chaque discussion référence, le script génère :
			A. Heatmap intra-discussion

				visualisation des similarités internes,
				palette à seuils,
				annotation des scores.

			B. Barres horizontales inter-discussion

				comparaison des moyennes Top‑K,
				seuil de cluster = moyenne intra-discussion,
				identification des discussions appartenant au cluster.

			C. CSV détaillé

			Contient :

				scores,
				top‑K utilisés,
				statistiques,
				cluster OUI/NON.

			D. JSON du cluster

	Liste des discussions dont la moyenne inter > moyenne intra.
	
	Étape 5 — Boucle automatique

			Le script :

				traite toutes les discussions une par une,
				génère un dossier resultats_analyse/res-<discussion> pour chacune,
				exporte tous les graphiques et fichiers.
    
    
Objectif du script

		Ce projet analyse automatiquement les similarités entre discussions audio en utilisant des embeddings vectoriels.
		Pour chaque discussion, il calcule :

			la cohérence interne (intra-discussion),
			la similarité avec toutes les autres (inter-discussion),
			un cluster de discussions proches.

		Il génère des visualisations, des CSV détaillés et un fichier JSON de cluster.
		
Pipeline du script

		Analyse du JSON pyannote
		Chargement des embeddings
		Calcul des centroïdes robustes (Top‑K)
		Analyse intra-discussion
		Analyse inter-discussion
		Génération des rapports
		Boucle automatique sur toutes les discussions

Structure des fichiers

			project/
			│--- similitude_trois_meilleurs_auto.py
			│--- segmentation_audios_nettoyes.json
			│--- data/
			│      |--- embeddings_nettoyes.csv
			│--- resultats_analyse/
			│      |---   res-<discussion>/
			│                 |---  01_vecteur_intra_<discussion>.png
			│                 |---  02_moyennes_horizontales_<discussion>.png
			│                 |---  rapport_similitudes_<discussion>.csv
			│                 |---  cluster_locuteur_<discussion>.json

Paramètres importants

    LOCUTEUR_CIBLE : locuteur analysé
    NB_TOP_SEGMENTS : segments utilisés pour le centroïde robuste
    SOURCE_DONNEES : CSV, PKL ou dossier .pt
    CIBLES_SPECIFIQUES : liste optionnelle pour filtrer les discussions

Exécution

		python similitude_trois_meilleurs_cluster.py
		
	fonctionnement automatique du corpus de discussions
	
	exemple de sortie terminal : 
		============================================================
		🔄 TRAITEMENT 61/70 | RÉFÉRENCE ACTUELLE : audio-1775033905.42249
		============================================================
		📊 Extraction et calculs Intra-discussion...
		 -> Moyenne intra-discussion calculée : 0.6099
		📊 Calculs Inter-discussions (69 cibles)...
		✅ Analyse terminée : 5 cibles dans le cluster de audio-1775033905.42249.
		📁 Dossier exporté : resultats_analyse/res-audio-1775033905.42249

		============================================================
		🔄 TRAITEMENT 62/70 | RÉFÉRENCE ACTUELLE : audio-1773398359.26843
		============================================================
		📊 Extraction et calculs Intra-discussion...
		 -> Moyenne intra-discussion calculée : 1.0000
		📊 Calculs Inter-discussions (69 cibles)...
		✅ Analyse terminée : 0 cibles dans le cluster de audio-1773398359.26843.
		📁 Dossier exporté : resultats_analyse/res-audio-1773398359.26843
		
	



