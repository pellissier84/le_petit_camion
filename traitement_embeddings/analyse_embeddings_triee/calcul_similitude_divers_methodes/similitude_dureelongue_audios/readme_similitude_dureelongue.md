Analyse des Segments Audio & Profils de Locuteurs

Readme du script : similitude_duree_wav.py

Objectifs du Script

    Identifier les discussions contenant un locuteur cible.
    Extraire les segments audio pertinents (durée > 2 sec).
    Charger les embeddings (formats .pt, .pkl, .csv).
    Construire un profil vocal de référence basé sur les k segments les plus longs.

    Calculer :

        Similitudes intra-discussion (référence vs autres segments).
        Similitudes inter-discussion (référence vs autres discussions).

    Générer :

        Un heatmap analytique.
        Un graphique horizontal comparatif.
        Un rapport CSV enrichi.
        
Structure Générale du Script

	1. Analyse du JSON Pyannote

		Fonction : analyser_discussions_et_durees()

			Parcourt le JSON contenant les segments audio.
			Filtre les segments du locuteur cible.
			Conserve la durée exacte de chaque segment.
			Identifie les discussions valides (≥ 2 segments du locuteur).
    
	2. Chargement des Embeddings

		Selon le format :

			.pt → chargement PyTorch
			.pkl → Kiwano ou PyTorch
			.csv / .tsv → parsing manuel

Fonctions principales :

		charger_segments_tries_pt()
		charger_embeddings_csv()
		filtrer_embeddings_memoire()
    
3. Calculs Mathématiques & Temporels

	Profil de référence

		Fonction : obtenir_moyenne_plus_longs()

			Sélectionne les k segments les plus longs.
			Calcule leur moyenne vectorielle → profil vocal.

	Intra-discussion

		Fonction : calculer_intra_discussion()

			Compare le profil aux autres segments de la même discussion.
			Produit un vecteur de similarités + une moyenne globale.

	Inter-discussion

		Fonction : calculer_inter_discussion()

			Compare le profil de référence à celui d’une autre discussion.
			Retourne un score unique de similarité.
    
4. Visualisations & Export CSV

	Fonction : generer_rapports()

			Heatmap des scores intra-discussion.

			Barres horizontales des scores inter-discussion.

			Export CSV : Structure du CSV

				|      Colonne             | Signification |
				| ------------------------ | ---------------------------------------------- |
				| **Type**                 | INTRA = comparaison interne à la discussion de référence ; 
											  INTER = comparaison avec une autre discussion 
											  exemple : "INTRA (Référence)", nom_ref, round(moyenne_intra, 4), ...|
				| **Discussion**           | Identifiant de la discussion analysée 
												Exemple : audio-1773398136.26807|
				| **Score Similitude**     | Similarité cosinus entre les profils vocaux (entre -1 et 1) 
												La moyenne des similarités cosines entre :    le profil de référence (moyenne des k segments les plus longs)
												tous les autres segments de la même discussion.|
				| **Nb Segments Evalués**  | Nombre total de segments utilisés pour la comparaison |
				| **ID du Profil**         | Les identifiants des k segments les plus longs utilisés pour construire le profil vocal de référence |
				| **Durée Profil**         | Somme des durées (en secondes) des *k* segments les plus longs |
				| **Score Min (Intra)**    | Min et max des similarités cosines entre :  le profil de référence et les autres segments de la même discussion
				| **Score Max**            | Score Min / Max : toujours N/A car en INTER il n’y a qu’un seul score.|
        
Organisation des Fichiers

			project/
			│
			├── segmentation_audios_nettoyes.json   # Segments Pyannote
			├── data/
			│   └── embeddings_nettoyes.csv         # Embeddings
			│
			├── resultats_analyse/                  # Graphiques + CSV générés
			│   ├── 01_vecteur_intra_*.png
			│   ├── 02_moyennes_horizontales_*.png
			│   └── rapport_similitudes_*.csv
			│
			└── similitude_duree_wav.py                  # Script principal

Commentaires Techniques Importants

    Les segments sont filtrés par durée (> 2 sec).
    Le profil vocal est basé sur les k segments les plus longs → plus stable.
    La similarité est calculée via cosine similarity.
    Le script supporte plusieurs formats d’embeddings.
    Les visualisations utilisent Seaborn + matplotlib.
