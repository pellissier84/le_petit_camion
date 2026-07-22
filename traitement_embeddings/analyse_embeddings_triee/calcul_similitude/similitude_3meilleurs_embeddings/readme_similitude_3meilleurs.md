# Analyse de Similitude Vocale : Méthode des Centroïdes (Top-K)

##  Présentation du projet

Ce script évalue la similarité vocale d'un locuteur cible à travers 
plusieurs enregistrements audio. Pour garantir une comparaison robuste et 
s'affranchir des anomalies (bruits de bouche, hésitations, erreurs de segmentation), 
ce programme utilise la méthode des **centroïdes**. 

Plutôt que d'évaluer tous les segments de manière égale, le script identifie mathématiquement 
les `K` segments les plus représentatifs d'une discussion (ceux dont l'embedding est le plus 
proche de la moyenne globale de la discussion) pour construire une "empreinte vocale" de référence pure.

##  Fonctionnement Mathématique

1. **Calcul du Centroïde :** 
		Pour une discussion donnée, le script calcule la moyenne de tous les vecteurs du locuteur.
		
2. **Élection du Top-K :** 
		Les `K` segments (par défaut 3)(peut etre modifié dans la partie : Exécution Principale au niveau "NB_TOP_SEGMENTS = 3" 
			Constante ajustable selon vos besoins de recherche)	ayant la plus haute similarité cosinus 
			avec ce centroïde sont sélectionnés. Leurs vecteurs sont moyennés pour créer l'empreinte finale du profil.
			
3. **Analyse Intra-discussion :** 
		Confronte l'empreinte parfaite (Top-K) au reste des segments (bruités) de la *même* discussion.
		
4. **Analyse Inter-discussion :** 
		Compare l'empreinte parfaite (Top-K) de la référence avec l'empreinte parfaite 
			(Top-K) d'une autre discussion.

##  Prérequis et Fichiers d'entrée

* **Fichier JSON (Pyannote) :** 
		Contient la segmentation temporelle (`start`, `end`, `speaker`). 
		Exemple : `segmentation_audios_nettoyes.json`.
		
* **Fichiers d'Embeddings :** 
	Les vecteurs mathématiques des segments. Le script supporte :
	  * Un fichier `.csv` ou `.tsv` unifié present dans "data/embeddinds_nettoyes.tsv" (recommandé)
			(issu du traitement et extraction des embeddings grace a Kiwano).
	  * Un dictionnaire compressé `.pkl` (via Kiwano).
	  * Un dossier contenant de multiples fichiers tenseurs `.pt`.

##  Données de Sortie (Dossier `resultats_analyse/`)

À chaque exécution, le script génère un rapport horodaté contenant :

1. **`01_vecteur_intra_[TIMESTAMP].png`** : 
		Une carte de chaleur 1D illustrant le score de chaque segment 
			écarté face au profil de référence.(il faut un minimum de 3 segments pour l'affichage).
			
2. **`02_moyennes_horizontales_[TIMESTAMP].png`** : 
		Un graphique en barres classant les discussions 
			cibles selon leur similarité avec la référence.
			
3. **`rapport_similitudes_[TIMESTAMP].csv`** : 
		Un tableau d'audit complet détaillant les scores 
			(moyenne, min, max), le nombre de segments évalués, et listant explicitement les identifiants exacts 
			des segments constituant les profils Top-K.
	
	description du csv.
	
	| Colonne CSV              | Signification                                     |  Ce que cela t’apprend |
	| ------------------------ | ------------------------------------------------- | ---------------------- |
	| **Date Exécution**       | Horodatage de l’analyse                           | Permet de tracer quand l’analyse a été faite |
	| **Locuteur Cible**       | Locuteur analysé                                  | Indique pour quel locuteur les profils sont construits |
	| **Type Analyse**         | INTRA ou INTER                                    | Permet de distinguer analyse interne et comparaison entre discussions |
	| **ID Discussion**        | Nom de la discussion                              | Identifie la source audio analysée |
	| **Nb Segments Total**    | Nombre de segments du locuteur dans la discussion | Indique la quantité de données disponibles pour cette discussion |
	| **ID du Profil (Top 3)** | Les 3 segments les plus proches du centroïde      | Ce sont les segments les plus “purs”, ceux qui définissent le profil vocal |
	| **Similitude Moyenne**   | Score de similarité cosinus                       | Mesure la proximité vocale entre profils |
	| **Score Min**            | Segment le moins proche du profil                 | Indique la variabilité interne de la discussion |
	| **Score Max**            | Segment le plus proche du profil                  | Indique le meilleur segment de la discussion |
	| **Écart-Type**           | Variabilité des similarités                       | Plus il est faible, plus la discussion est homogène |
	Rq: un minimum de 3 segments pour l'affichage des scores et écart type.

##  Utilisation

Renseignez vos chemins de fichiers dans la section `CONFIGURATION MANUELLE` en bas du script, puis lancez l'exécution :

		SOURCE_DONNEES = "data/embeddings_nettoyes.csv" 
		FICHIER_JSON_PYANNOTE = "segmentation_audios_nettoyes.json"
		LOCUTEUR_CIBLE = "operateur"
		NB_TOP_SEGMENTS = 3 # Constante ajustable selon vos besoins de recherche
		
Lancement : 

	python similitude_trois_meilleurs_2.py
	
	affichage des discussions avec un minimum de 2 segments , avec un numero d'ordre, a indiqué par la suite pour choisir 
	la réference.
