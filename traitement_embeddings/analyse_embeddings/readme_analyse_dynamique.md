# 🎙️ Analyse Dynamique des Similitudes Cosinus (Speaker Verification)

	Ce pipeline automatisé permet d'évaluer la robustesse de la signature vocale d'un locuteur 
	à travers plusieurs discussions (enregistrements SDIS/Nexsis). Il compare la variabilité 
	interne d'un appel (Intra-discussion) avec la dérive biométrique face à d'autres appels (Inter-discussions).

## ✨ Fonctionnalités
	- **Mode Hybride Automatique :** Supporte la lecture de fichiers individuels `.pt` (PyTorch) ou 
		le chargement natif de corpus compressés `.pkl` via **Kiwano**.
	- **Sélection Intelligente :** Analyse un fichier de segmentation Pyannote (`.json`) 
		pour filtrer les discussions vides et propose un menu interactif pour choisir la discussion de référence.
	- **Rigueur Mathématique :** Utilise le calcul de similarité cosinus natif de PyTorch 
		(`torch.nn.functional`) avec produit cartésien complet pour comparer les appels sans écraser les nuances vocales.
	- **Visualisations Avancées :**
		- Matrice triangulaire (Heatmap) avec paliers stricts pour l'analyse intra-discussion.
		- Bâtons horizontaux triés de -1 à 1 pour la comparaison inter-discussions.

## 🛠️ Prérequis et Installation
	Assurez-vous d'avoir installé les bibliothèques suivantes dans votre environnement virtuel :

	pip install torch numpy matplotlib seaborn pandas

	(Optionnel) Si vous utilisez des fichiers .pkl, le module interne de M. Rouvier (kiwano) doit 
	être accessible dans votre environnement Python

Structure des Données Attendue

	Le script s'attend à lire les données selon l'un de ces deux formats :

		Format Dossiers (.pt) : data/embeddings_corpus/<id_discussion>/<nom_segment>.pt

		Format Kiwano (.pkl) : Un fichier unique data/xvectors.pkl généré par Kiwano.
		
		Format Texte Personnalisé (.csv / .tsv / .txt)
			Le script prend également en charge les exports d'embeddings stockés dans un fichier texte unique. 
			Pour être lu correctement par le parseur, chaque ligne du fichier doit respecter la syntaxe stricte suivante : 
			**l'identifiant complet du segment**, un séparateur **"pipe" (`|`)**, puis **les valeurs numériques** séparées par 
			des espaces.

**Exemple de ligne valide :**
`audio-1775032164.41898_operateur_41|0.128763 0.079515 -0.153897 0.525902 ...`

	Vous devez également disposer d'un fichier de segmentation généré par Pyannote/WhisperX (ex: segmentation_audios_Nexsis.json).

Utilisation

    Ouvrez le fichier analyse_dynamique.py et modifiez la section CONFIGURATION tout en bas selon vos chemins réels :
		SOURCE_DONNEES = "data/embeddings_corpus/" # ou "data/corpus.pkl"
		FICHIER_JSON_PYANNOTE = "segmentation_audios_Nexsis.json"
		LOCUTEUR_CIBLE = "operateur"
		
Structure du dossier

dossier
	|--- data
	|      |-- embeddings_corpus
	|      |          |--- (embeddings des wav au format .pt)
	|      |--- embeddings.csv (embeddings issu de kiwano)
	|
	|--- resultats_analyse
	|       |--- matrice.png
	|       |--- moyenne.png
	|       |--- rapport.csv
	|--- analyse_dynamique.py
	|--- segmentation_audios_Nexsis.json
	|--- readme_analyse_dynamique.md
	
Lancez le script : python analyse_dynamique.py

Sorties Générées

	Les résultats sont sauvegardés dans un dossier horodaté resultats_analyse/ :

		01_matrice_intra_[date].png : Évalue la stabilité de la voix du locuteur pendant l'appel de référence.

		02_moyennes_horizontales_[date].png : Compare le locuteur de référence à ses apparitions présumées dans les autres appels.

		rapport_similitudes_[date].csv : Tableau de synthèse brut pour l'équipe de recherche.

le calcul de la similitude inter-discussions (entre la référence et une autre discussion) ne se fait pas en comparant 
	un "résumé" de l'appel A avec un "résumé" de l'appel B.
	une méthode beaucoup plus fine et robuste qui repose sur le produit cartésien des similarités cosinus. 
	L'idée est de croiser absolument chaque instant de la première discussion avec chaque instant de la seconde.
	
étape par étape :

Les données de départ (Les Embeddings)

Pour rappel, un embedding est un grand tableau de nombres (souvent 192 ou 512 dimensions) qui représente 
		la "signature vocale" d'un segment de quelques secondes.
    • Imaginons que la Discussion Référence (A) possède 3 segments de l'opérateur : R1, R2, R3.
    • Imaginons que la Discussion Cible (B) possède 2 segments de l'opérateur : C1, C2.
Le Produit Cartésien (Toutes les paires possibles)
Plutôt que de faire une moyenne de R1, R2, R3 pour la comparer à la moyenne de C1, C2 
		(ce qui écraserait les nuances de la voix), le script crée toutes les paires possibles entre les deux discussions.
La fonction product() de la bibliothèque itertools génère ce croisement :
    1. R1 est comparé à C1
    2. R1 est comparé à C2
    3. R2 est comparé à C1
    4. R2 est comparé à C2
    5. R3 est comparé à C1
    6. R3 est comparé à C2
Dans cet exemple, pour 3 segments d'un côté et 2 de l'autre, la machine effectue 6 comparaisons (3 x 2).
