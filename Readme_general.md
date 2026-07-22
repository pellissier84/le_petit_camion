readme des depots Github

pipeline de travail sur les audios discussions des appels de secours

description du travail sur les enregistrements des discussions :

fichiers necessaires :
	des enregistrements de discussion au format .wav
	des transcriptions automatiques recueillies par le traitement de Whisper au format .json;
	des diarisations automatiques (separation locuteur) recueillies par le traitement par le traitement de Pyannote au format .json;
	des embeddings (vecteur 256 dimensions) des segments de discussion par locuteur recueillies par traitement de Kiwano au format .csv

1) "script_de_conversion"
	Pour avoir des references essentielles sur ces enregistrements
	utilisation de Elan 7.1 qui permet à partir d'un enregistrement .wav de pouvoir le sectionner en locuteurs et 
	en faire les transcriptions manuellement de ce qui est dit.
	les fichiers sont enregistres au format .eaf et peuvent etre segmentés et regroupés par locuteurs.

2) "NER"
	recherche des mots importants metiers dans les transcriptions eaf et wav, et les affiche.
	Calcul et mise en evidence des differences d'appréciation par un WER, Word Error Rate, ou Taux d'erreur de mots, 
	est une unité de mesure classique pour mesurer les performances d'un système de reconnaissance vocale.

3) "travail_segmentation_pyannote"
	Suite aux travaux par des systemes automatiques de reconnaissances de la parole.
	Whisper : est un modèle d'apprentissage automatique pour la reconnaissance et la transcription vocales,
	Pyannote :  une solution de diarisation permettant d’identifier les intervenants dans un fichier audio.
	Les technologies de transcription automatique et de diarisation permettent une exploitation plus avancée des données audio. 
	Whisper et Pyannote, combinés, offrent des transcriptions précises et segmentées par interlocuteur, 
	facilitant leur utilisation dans divers contextes.
Le premier sous dossier permet de faire des recherches et visualiser le contenu du json issu de pyannote, 
	et comparaison par rapport aux transcriptions manuelles Elan.
Le deuxieme sous dossier fonctionne pour extraire des mots metiers des transcriptions whisper et de la diarisation Pyannote.

4) "contexte_mot_metier" script optionnel de recherche
	Essai de travail sur la contextualisation des mots metiers dans une discussion. Mais les prises de parole trop courtes des locuteurs 
	posent un probleme d'evaluation des contextes.
	
5) "identification-locuteur"
	A partir d'un decoupage d'un enregistrement de discussion, on extrait des embeddings (Un embedding est une représentation vectorielle dense 
	d'une entité (mot, phrase, document, image, etc.) dans un espace vectoriel continu de dimension réduite 
	(typiquement 128, 256, 512, 768, 1536 dimensions), où la proximité géométrique reflète la similarité sémantique ou contextuelle entre les entités). 
	Les embeddings sont issus de Speechbrain (un toolkit tout-en-un liant PyTorch et le traitement automatique de la parole. 
		Basé sur le succès de son prototype PyTorch-Kaldi, c'est un outil unique, flexible et surtout facile à prendre en main, 
		qui puisse être utiliser pour rapidement développer des systèmes état de l’art pour la parole.)
	On prend un locuteur qui sera la référence, on le compare à un corpus pour identifier la reference parmi cet corpus.
	Utilisation de plusieurs methodes de scoring, pour extraire et reconnaitre un locuteur particulier parmi un corpus.

6) "Traitement_embeddings"
			Traitement quasi identique au precedent "identification-locuteur" mais plus specifique, utilisation de Kiwano.
			A partir d'un fichier regroupant les embeddings extraits avec l'outil Kiwano au format tsv (csv), travail pour identifier un locuteur reference
			La sortie du traitement, fournit une matrice de comparaison intra-discussion, un graphique batonnet inter discussion du corpus de discussion.
			
			
	analyse embeddings simple puis trié puis amelioré par certaines techniques : 
	travail sur la reconnaissance de locuteurs
		l'analyse se fait d'un facon generale, puis sur des moyens pour ameliorer les resultats sur plusieurs options dont 2 ont été retenus
		Dans le dossier "calcul_similitude" sont regroupés traitement simple et brut, les 3 meilleurs embeddings, la duree des segments les plus longs,
		mais aussi une direction non retenu centroide, agregation et ponderation.	
	
7) "amelioration_qualite_wav" : script optionnel de recherche 
	Script pour ameliorer les wav originaux (travail complementaire a l'extraction des embeddings)
	Traitement des fichiers, pour obtenir des wav pouvant ameliorer l'extraction des embeddings et des reconnaissance de locuteur.

8) "extraction_embeddings_speechbrain"	: script optionnel de recherche
	Script de travail personnel pour extraire des embeddings pour appliquer mes scripts (travail complementaire)
	Traitement des audios format wav originaux complets, calcul des embeddings sur chaque segmentation des discussions.


Detail du depot et des differents dossiers

## Dossier "script_de_conversion"
--- contient

le script : "convertir_elan.py"

		Script pour convertir l'export ELAN vers le format Audacity
		# Ce script lit un fichier texte exporté depuis ELAN, extrait les annotations et les regroupe par intervenant (locuteur).
		# Chaque intervenant aura son propre fichier de sortie au format Audacity, contenant les segments temporels correspondants.
		# Prérequis : Python 3 et le fichier d'entrée doit être au format texte (exporté depuis ELAN) avec des colon
		# nes séparées par des tabulations.
		# Usage : python convertir_elan.py mon_fichier.txt
		# Version : 1 fichier par intervenant
		
le script : "extraire_elan_diff_tier.py"

		Extraction automatique des intervenants depuis un fichier ELAN (.eaf)
		- Menu interactif pour le choix des tiers
		- SAUVEGARDE des petits fichiers .wav (segments) avec le numéro de l'appel
		- CONCATÉNATION en un gros fichier .wav par locuteur

		prerequis : python3 et ffmpeg
		# Ce script lit un fichier .eaf, affiche les tiers disponibles, et permet à l'utilisateur de choisir lesquels extraire.
		# Il offre aussi l'option de séparer les annotations par valeur (ex: pour un tier "Ac_ev", créer des fichiers distincts pour "und", "bruit", "musique", etc.).
		# Les segments audio correspondants sont extraits du fichier source et concaténés en un seul fichier par tier/valeur.
		# Les fichiers résultants sont nommés de manière claire (ex: "Ac_ev_und.wav", "Ac_ev_bruit.wav", etc.).
		# conda activate audio_env
		# commande de lancement : python extraire_elan.py mon_corpus.eaf enregistrement.wav
		# exemple python extraire_elan_diff_tier.py audio_elan_3540.eaf audio_elan_3540.wav
		# exemple python extraire_elan_diff_tier.py audio-1775033540.32214.eaf audio-1775033540.32214.wav
		
le script : "extraire_wav_complet.py"

		Extraction automatique des intervenants depuis un fichier json (pyannote)
		# Les segments audio correspondants sont extraits du fichier source et concaténés en un seul fichier par tier/valeur.
		# Les fichiers résultants sont nommés de manière claire
		- Menu interactif pour le choix des fichiers wav
		- SAUVEGARDE des petits fichiers .wav (segments) avec le numéro de l'appel (segments_extraits)
		- CONCATÉNATION en un gros fichier .wav par locuteur (segments_complets_extraits) avec ffmpeg

		requis : 

		DOSSIER_WAV = "mes_audio_wav" contenant un corpus
		FICHIER_JSON = "segmentation_audios_Nexsis.json" resultant de Pyannote
		DOSSIER_SEGMENTS = "segments_extraits"
		DOSSIER_COMPLETS = "segments_complets_extraits
		
	et les trois readme de ces scripts
	
## Dossier "NER"
	Il contient trois sous dossiers evaluate_transcription v2, v3 et final
	Le plus aboutit est evaluate_transcription_Version_Finale

Ce dernier dossier contient :

le fichier : "communes-haute-garonne.geojson.json"

		nomenclature de toutes les communes du departement 31.
		Rq : Il existe sur le site du gouvernement des fichiers contenant toutes les adresses de chaque commune de France.
		Ce fichier est nécessaire pour trouves les communes du 31 dans les appels.
		
le fichier : "patterns-sdis-cisu.jsonl"

		Il regroupe en entités nommées touts les elements du CISU, c'est à dire les feuilles risques et menaces; 
		nature et faits; motif medico-secouristes et types de lieu organisés pour etre compatibles avec 
		l'EntityRuler de spaCy (IA spécialisé sur le NER)
		
le script : "evaluate_wer_metier_complet_V1.py"

		# evaluation et recherche des mots metiers transcription elan (eaf) versus whisper(json)
		# exemple de commande : python evaluate_wer_metier_complet_V1.py --eaf "audio-1775031826.41778.eaf" --json "audio-1775031826.41778.json"

		# calcul et visuel sur les evaluations de whisper
		# utilise : EntityRuler de spacy avec patterns_sdis_cisu.jsonl (mots issus du CISU)
		# utilise : Regex des entités nommées (mots métiers)
		# utilise : gazetteer recuperation des communes du departement 31
		# utilise : module de detection des entités nommés NER par spaCy
		
		Il est accompagné par son readme et des exemples de sortie
		
## dossier "travail_segmentation_pyannote"
Il contient deux sous dossiers.

### Sous dossier "travail_source"
travail sur le fichier segmentation_audios_Nexsis.json (sensible non present)

le script : "appy.py"

		# visualisation des wav diariser par pyannote
		# nom du fichier : segmentation_audios_Nexsis.json
		# installation nécessaire : pip install streamlit matplotlib
		# utilisation du script app.py dans le meme dossier que le json
		# depuis un terminal : streamlit run app.py
		# Une page web va s'ouvrir automatiquement dans votre navigateur. 
		# Sur la gauche, vous aurez un menu déroulant listant vos differents fichiers. 
		# Dès que vous en sélectionnez un, le chronogramme et les temps de parole s'affichent instantanément, 
		# proprement, et sans inonder votre ordinateur de centaines d'images.
		
le script : "lister_fichiers.py"

		# Script : lister les fichiers audio présents dans un JSON Pyannote
		# commande : python lister_fichiers.py
		# Ce script lit un fichier JSON contenant une clé "files" et
		# extrait tous les noms de fichiers audio associés à cette clé.
		# Il affiche la liste dans la console et la sauvegarde dans un
		# fichier texte : liste_fichiers_nexsis.txt
		
le script : "decoupage_audio_liste.py"

		# Script de découpage audio par locuteur
		# ---------------------------------------
		# Ce script lit un fichier JSON issu d'une diarisation (ex : pyannote)
		# et découpe automatiquement un fichier WAV en segments individuels
		# pour chaque locuteur, avec une numérotation séquentielle.
		# decoupage des wav de la liste json issu de pyannote
		# utilisation du fichier sorti précédemment liste_fichiers_nexsis.txt
		# decoupage en locuteur et par intervention et numero d'ordre
		# Format du nom : audio-XXXXX_locuteur_ordre.wav
		# commande : python decoupage_audio_liste.py
		# sortie dans un dossier de tous les fichiers decoupés  : locuteurs_extraits
		# requis : python 3.7 min, pydub , ffmpeg
	Rq ; pyannote n'a pas nommé tous les locuteurs, amis des chiffres pour l'ordre
		
le script : "decoupage_audio_mappage.py"

		# respect des noms originels des fichiers wav
		# extractions des segments wav d'un locuteur
		# sans concatenation de wav
		# meme fonctionnement que decoupage_audio_liste.py sans le mapping des locuteurs


		MAPPING_LOCUTEURS = {
			"operateur": "OP-SDIS1",
			"0": "requerant0",
			"1": "requerant1",
			"2": "requerant2",
			"3": "requerant3",
			"4": "requerant4"
		}

		DOSSIER_WAV = "mes_audios_wav"
		FICHIER_JSON = "segmentation_audios_Nexsis.json" 
		DOSSIER_SORTIE = "locuteurs_extraits"
		
le script : comparaison_diarisation.py
		# comparaison visuelle des references eaf par rapport au wav decoupes par pyannote
		# affichage des locuteurs
		# en utilisant : segmentation_audios_Nexsis.json
		# les eaf et wav qui seront comparer ont meme nom
		# ex : audio-1775031826.41778
		bibliotheques  :  pympi et matplotlib.pyplot
		
le script : comparaison_metrique_diarisation_V1.py
		# rapport metrique entre eaf et wav decoupé suivant les locuteurs de pyannote
		# les eaf (meme nom pour les .wav)
		# ex : audio-1775031826.41778.eaf
		# pour cet audio wav coupé a 4mn, et enlevement du tier Ac_ev

		# fichier des decoupes par pyannote de 116 wav :
		# segmentation_audios_Nexsis.json
		
	Les readme de chaque script
	
### Sous dossier : "travail_audio_whisper_pyannote"
Plus de reference aux annotations Elan travail a partir des wav , des json whisper et du json Pyannote

le script : "lister_fichiers.py"

		present dans un autre dossier
		# Script : lister les fichiers audio présents dans un JSON Pyannote
		# commande : python lister_fichiers.py
		# Ce script lit un fichier JSON contenant une clé "files" et
		# extrait tous les noms de fichiers audio associés à cette clé.
		# Il affiche la liste dans la console et la sauvegarde dans un
		# fichier texte : liste_fichiers_nexsis.txt

le script : "extract_mots_metiers.py"

		# recherche des mots metiers
		# a partir du json whisper et json pyannote
		# par rapport aux audio wav
		# audio-1775031968.41843
		# audio-1775033540.32214
		# audio-1775031826.41778
		# recherche par regex, ner ,EntityRuler (patterns_sdis_CISU), gazetteer communes
		# affichage résultat sur ecran et sous forme de json

		"""
		Script d'extraction d'entités et de mots métiers pour les appels d'urgence (CISU/SDIS).
		Combine une ontologie par expressions régulières (Regex) et la reconnaissance 
		d'entités nommées (NER) via spaCy pour repérer des concepts critiques dans des 
		transcriptions WhisperX, puis attribue ces concepts aux locuteurs via Pyannote.

		Exemple d’utilisation

			Lancer le script :
			python extrac_mots_metier.py
			
			Le programme demande :
			============================================================
			 Fichiers disponibles dans 'mes_audio' :
			============================================================
			 [1] audio-1775031826.41778.json
			 [2] audio-1775031968.41843.json
			 [3] audio-1775033540.32214.json
			 [0] Quitter le programme
			============================================================

		Entrez le numéro du fichier à analyser :
		

Ces Scripts sont accompagnés d'un readme et d'un guide montrant les resultats.
Présence d'un fichier patterns_sdis_cisu.jsonl nécessaire au script "extract_mots_metiers.py", manque le json resultant de Pyannote


		
## Le dossier "contexte_mot_metier"

les fichiers json et jsonl, recueil des mots metiers cisu et des communes 31.

le readme du script

le script : "evaluate_wer_metier_contexte_V2_optimise.py"

Évaluation des mots métier avec analyse contextuelle
Nom du fichier : evaluate_wer_metier_contexte_V2_optimise.py

DESCRIPTION ARCHITECTURALE :
		Ce script compare une transcription automatique (WhisperX) à une référence humaine (ELAN)
		pour évaluer la reconnaissance d'un vocabulaire métier spécifique (SDIS/CISU).
		Il utilise une approche hybride à 3 niveaux pour l'extraction :
		1. Règles strictes (Expressions Régulières - Regex)
		2. Dictionnaire métier (spaCy EntityRuler via le fichier JSONL)
		3. Intelligence Artificielle générique (spaCy NER)
		Il intègre ensuite une analyse du contexte (mots environnants) pour confirmer
		ou infirmer la pertinence des termes détectés.

		Utilisation :
		python evaluate_wer_metier_contexte_V2_optimise.py --eaf "audio-1775031826.41778.eaf" --json "audio-1775031826.41778.json"

## Le dossier "identification-locuteur"
essai , premier script non abouti, travail en cours
deux sous dossiers travaillant sur la reconnaissance du locuteur

### Sous dossier : lecture_embeddings-locuteur
	Utilisation des scripts issus du depot Github Kiwano (https://github.com/mrouvier/kiwano/tree/main/recipes/speaker_verification)
	(compute_asnorm.py; compute_adnorm.py; compute_snorm.py; compute_cosine.py; base.py (kiwano/embeddings))
	Recherche interactive d'un locuteur dans un corpus d'embeddings.

le script : "search-with_kiwano1.py"

	Ce script permet de :
		- Charger un fichier d'embeddings (format .pt ou .pkl Kiwano) comme référence
		- Comparer cette référence à un corpus d'embeddings
		- Utiliser différentes méthodes de scoring : cosinus simple, S-Norm, AS-Norm, AD-Norm
		- Afficher et sauvegarder les résultats avec un seuil personnalisable

	Cosinus simple			Similarité cosinus brute entre l'embedding de référence           	compute_cosine.py
								et chaque embedding du corpus.	
		S-Norm	            Normalisation symétrique : normalise le score en utilisant       	compute_snorm.py
								la moyenne et l'écart-type des scores contre un ensemble 
								d'impostors.	
		AS-Norm	            Adaptive S-Norm : sélectionne les *k* meilleurs impostors          	compute_asnorm.py
								pour le calcul de la moyenne et de l'écart-type.	
		AD-Norm	            Adaptive Domain Normalization : soustrait la moyenne 
								des *k* meilleurs impostors au niveau des embeddings avant       compute_adnorm.py
								de calculer la similarité.

le script : "base.py" (copie depuis kiwano) pour lire et ecrire les fichiers avec l'extension .pkl

Structure des fichiers

	mon_dossier/
		├── search_with_kiwano1.py # Script principal
		├── base.py # Module Kiwano (optionnel pour .pkl)
		├── embeddings_reference.pt # Fichier d'embeddings de référence
		└── embeddings_corpus.pt # Fichier d'embeddings du corpus
		
le Readme du script principal


### Sous dossier : lecture_wav_recherche_locuteur

Dédié à l'extraction d'embeddings vocaux et à la classification de
locuteurs. Conçu pour être robuste face aux bruits de fond, il est particulièrement adapté au traitement de
corpus d'appels opérationnels

le script : "tri_ameliore_embeddings1.py"

	Le script fournit :

		 un prétraitement audio avancé,
		 une détection de parole (VAD) robuste,
		 plusieurs modes d’extraction d’embeddings,
		 une sélection automatique de la meilleure référence,
		 une classification intra- ou inter-discussions,
		 des visualisations complètes,
		 une sauvegarde structurée des résultats.
		 
	Structure des fichiers

			mon_dossier/
				├── tri_ameliore_embeddings1.py # Script principal
				├── readme
				├── mes_audios
				|     |-- reference
				|     |-- test
				|
				|-- resultats
				└── pretrained_model
				
le Readme du script principal et analyse des embeddings (sortie ecran du traitement)

## Dossier "Traitement_embeddings"

### Sous dossier : analyse_embeddings
		traitement des embeddings suivant leur format, format csv issu de kiwano, 
		fichier necessaire le fichier json issu de pyannote
		avec 3 sorties : intra discussion sous forme de matrice
						inter discussion sous batonnets de moyennes
						rapport .csv
					 
		script : "analyse_dynamique.py"
		
			choix des discussions des locuteurs pour la reference, et comparaison des similitudes cosinus des embeddings.
				
		readme_analyse_dynamique

		data : corpus des embeddings, issu de speechbrain format .pt ou de Kiwano format .csv

		resultats_analyse : production des resultats visuels et fichiers de données.
					 
### Sous dossier "Analyse_unitaire"

		Amelioration du script "extract_embeddings.py" fourni par un collaborateur en : "extract_embedding_individuel.py" . 
		On fait le choix du locuteur, de la discussions pour en sorti l'embedding ciblé.
		Dans le meme dossier, le fichier csv des embeddings extraits, et le readme de "extract_embedding_indiviuel.py"

### sous dossier "Analyse_embeddings_triee"

		TRAVAIL PRINCIPAL de reconnaissance des locuteurs parmi un corpus.
		
		Option d'amélioration des analyses d'embeddings en enlevant les segments d'un locuteur d'une durée inferieure ou à égale à 2s, 
		des segments vides. On commence par un tri dans les segments depuis le csv issu de Kiwano a partir du json issu de pyannote.

#### premier dossier : "tri_embeddings_segmentation"

	Des outils pour triés les segments audios et donc les embeddings, suivant leur durée >2s, a partir du json pyannote et 
	calqué sur le fichier des embeddings csv issu de Kiwano. On obtient des fichiers json et csv nettoyes. 

	script "filtre_json_csv.py" : outil de tri

	script "verifier_temps_parole.py" : liste par discussions par locuteur par segment la durée des segments 
		(verification visuelle du temps superieur a 2s).
		
	readme et fichier necessaire (embeddings.csv Kiwano, segmentation_audios_Nexsis.json Pyannote)
	
#### deuxieme dossier : "calcul_similitude" (Travail Principal)

 dans ce dossier, etude et amelioration des resultats des moyennes et similitudes
 
		## dossier "similitude_par_moyenne_simple"
				script : "analyse_dynamique.py"
				
				Ce script permet d’analyser la similarité entre des segments audio appartenant à différentes discussions, 
					en utilisant des embeddings (PyTorch, PKL ou CSV).
					Il s’appuie sur un fichier JSON issu de Pyannote pour identifier les discussions valides et sur un fichier 
					d’embeddings pour effectuer les calculs.
					
					Deux types de calculs sont effectués :

						Intra-discussion : similarité entre les segments d’une même discussion
						Inter-discussion : similarité entre une discussion de référence et les autres discussions.
					
		## dossier "similitude_dureelongue_audios"
				recherche d'amélioration par tri des segments les plus longs.
				script "similitude_duree_wav.py"
				
				Ce script permet d'identifier les discussions contenant un locuteur cible.
					Extraire les segments audio pertinents (durée > 2 sec).
					Charger les embeddings (formats .pt, .pkl, .csv).
					Construire un profil vocal de référence basé sur les k segments les plus longs.
				
		## dossier "similitude_3meilleurs_embeddings"
				recherche d'amélioration par tri des segments en isole les 'k' meilleurs segments (les plus proches du centroïde global) 
					et calcule leur moyenne pour former le profil de référence robuste.
					script : "similitude_trois_meilleurs.py"
					
				Ce script, plutôt que d'évaluer tous les segments de manière égale,  identifie mathématiquement 
					les `K` segments les plus représentatifs d'une discussion (ceux dont l'embedding est le plus 
					proche de la moyenne globale de la discussion) pour construire une "empreinte vocale" de référence pure.
				
		## dossier "similitude_centroide_ponderation" (recherche additionnelle)
				recherche par une autre option (non retenu pour l'etude) approche par calcul a partir du centroide, ou par une ponderation de la durée.
				script : "similitude_centroide_ponderation.py"
				
				Ce script calcule la similarité intra-discussion en utilisant l'agrégation.
					- Calcule la similarité des deux moitiés (Pairs vs Impairs)
					- Calcule la similarité de chaque segment contre le profil global
					- comparaison segment → profil global
					- split-half (pairs vs impairs)
				Ce script transforme plusieurs segments en un seul vecteur représentatif.
					- centroide : moyenne simple
					- ponderation : moyenne pondérée par la durée.
	

## dossier "extraction_embeddings_speechbrain" (recherche additionnelle)

	script "extraction_embeddings_speechbrain.py"
			Ce script automatise l'extraction de signatures vocales (embeddings) à partir de fichiers audio bruts (`.wav`) 
			en s'appuyant sur un fichier de segmentation préalable (diarisation). Il utilise le modèle d'état de l'art **ECAPA-TDNN** 
			fourni par SpeechBrain.
	readme du script

## dossier "amelioration_qualite_wav" (recherche additionnelle)

	script "pipeline_amelioration_wav.py"
			Ce projet propose un pipeline automatisé de traitement audio (Batch Processing) conçu pour préparer et 
			nettoyer des enregistrements WAV en vue d'une tâche de diarisation ou d'identification du locuteur.

			Contrairement à un script de nettoyage classique, cet outil intègre un **mécanisme de décision scientifique (Gatekeeper)** : 
			il s'assure mathématiquement que le nettoyage par IA n'altère pas le timbre vocal (distorsion) avant de valider 
			la sauvegarde du fichier.
	readme du script


