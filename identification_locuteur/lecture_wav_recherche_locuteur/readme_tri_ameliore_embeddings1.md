Readme 

nom du script : tri_ameliore_embeddings1.py


Système de Classification et Diarisation de Locuteurs (SpeechBrain)

Description
Ce document accompagne le script Python dédié à l'extraction d'embeddings vocaux et à la classification de
locuteurs. Conçu pour être robuste face aux bruits de fond, il est particulièrement adapté au traitement de
corpus d'appels opérationnels (ex: flux impliquant des requérants, des opérateurs SDIS, et des intervenants
sur le terrain).
Le pipeline s'appuie sur l'architecture ECAPA-TDNN de SpeechBrain pour capturer les empreintes vocales,
et intègre des algorithmes de détection d'activité vocale (VAD) et de filtrage acoustique.
Capacité de Traitement : Le script est dimensionné pour analyser de larges volumes de
données de manière automatisée. Il est capable de traiter et de croiser efficacement un
corpus complet (ex: 116 fichiers d'enregistrements) en une seule passe, avec un affichage
étendu des résultats pour faciliter la vue d'ensemble du projet.

Ce projet propose un pipeline complet pour l’identification de locuteurs dans des corpus d’appels, en s’appuyant sur SpeechBrain (ECAPA-TDNN) et une série d’améliorations acoustiques adaptées aux environnements bruités (radio, téléphonie, enregistrements dégradés).

Le script fournit :

     un prétraitement audio avancé,
     une détection de parole (VAD) robuste,
     plusieurs modes d’extraction d’embeddings,
     une sélection automatique de la meilleure référence,
     une classification intra- ou inter-discussions,
     des visualisations complètes,
     une sauvegarde structurée des résultats.
     
Prétraitement audio amélioré

		Le script applique plusieurs étapes pour nettoyer les signaux téléphoniques :

			Normalisation RMS
			Filtre passe‑haut 80 Hz
			Préaccentuation (0.97)
			Resampling automatique en 16 kHz

			Exemple tiré du script :
			« Prétraitement du signal audio pour nettoyer les bruits de fond inhérents aux communications (téléphoniques ou radio). » 
			    
		Détection de parole (VAD)

		Un VAD basé sur l’énergie permet d’isoler les segments utiles et d’éviter les silences.
			« Détecteur d'Activité Vocale (VAD) basique par énergie. »
			
Extraction d’embeddings (SpeechBrain ECAPA)

		Plusieurs modes sont disponibles :

			vad : sélection de la zone la plus dense en parole
			top_k : frames les plus énergétiques
			weighted : pondération par énergie
			standard : extraction classique
			
Sélection automatique de la référence

		Pour chaque discussion, le script choisit le meilleur fichier de référence en combinant :

			qualité acoustique (SNR, RMS, durée, speech ratio)
			similarité au centroïde des embeddings
			« Sélectionne dynamiquement le meilleur extrait audio à utiliser comme référence. »
			
Classification intra-discussion

		Chaque fichier est comparé à la référence via un score de similarité (produit scalaire).

		Le script :
			exclut automatiquement les fichiers invalides (trop courts, trop silencieux, mauvaise qualité)
			calcule un seuil adaptatif basé sur la distribution des scores
			génère un tableau détaillé des décisions
			multi_frames : moyenne de plusieurs embeddings pour plus de robustesse	
			
## Structure des fichiers

	mon_dossier/
		├── tri_ameliore_embeddings1.py # Script principal
		├── readme
		├── mes_audios
		|     |-- reference
		|     |-- test
		|
		|-- resultats
		└── pretrained_model
				
			
Classification inter-discussions

		Permet de tester une référence sur tout le corpus, pour analyser :

			la cohérence des locuteurs
			les confusions entre discussions
			les performances globales du système
			
Visualisations

		Le script génère un dashboard complet :

			histogrammes
			scatter plots
			boxplots
			heatmaps de similarité
			statistiques par type de locuteur

Sauvegarde des résultats

		Le système génère des visualisations analytiques complètes (matrices de similarité, boxplots des
		distributions, analyses croisées taille/score) via Matplotlib. Il sauvegarde également l'ensemble des résultats
		de manière structurée dans un sous-dossier resultats/ comprenant : rapports textuels de session,
		tableaux CSV récapitulatifs, et tenseurs PyTorch (.pt) contenant les embeddings extraits pour une intégration
		ultérieure.

		Le système exporte :

			embeddings (.pt)
			tableaux de résultats (.csv)
			rapport texte (.txt)
			référence sélectionnée (.pt)			

			
Prérequis et Dépendances

		Assurez-vous de disposer d'un environnement Python avec les bibliothèques suivantes :
		python 3.8+
		pip install torch torchaudio speechbrain numpy scipy matplotlib pandas


Architecture du Pipeline

	Prétraitement Audio : Normalisation RMS, filtrage passe-haut (80Hz pour isoler les bruits de ligne
	téléphonique) et préaccentuation pour clarifier les fréquences vocales.
	Détection de Parole (VAD) : Analyse adaptative de l'énergie pour ne conserver que les segments
	contenant de la voix utile.
	Extraction d'Embeddings : Génération de vecteurs caractéristiques via 4 modes distincts : vad,
	top_k, weighted, ou multi_frames.
	Sélection de Référence Optimisée : Identification automatique du meilleur échantillon audio basé sur
	des scores de qualité (rapport signal-bruit, durée de parole continue).
	Classification & Diarisation : Calcul de la similarité cosinus couplé à un seuil adaptatif (basé sur
	l'écart-type des scores) pour affirmer si deux pistes appartiennent au même locuteur.

	Chargement modèle SpeechBrain
			↓
	Prétraitement audio
			↓
	Détection de parole (VAD)
			↓
	Extraction embeddings (4 modes)
			↓
	Sélection de la meilleure référence
			↓
	Classification (intra ou inter-discussions)
			↓
	Visualisation des résultats
			↓
	Sauvegarde (CSV, PT, TXT)


Utilisation

	Placez vos fichiers audio (au format .wav) dans le dossier cible (par défaut mes_audios/test) et lancez le
	script depuis votre terminal :
	python tri_ameliore_embeddings1.py

Périmètres d'analyse disponibles :

	Mode               |     Description
	Intra-discussion   | Vérifie la cohérence des locuteurs au sein d'une seule et même conversation.
	Inter-discussions  | Recherche l'occurrence d'un locuteur spécifique à travers l'intégralité du corpus. 
					   |   L'affichage console est paramétré pour lister jusqu'à 60 correspondances.
                  


