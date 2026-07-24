README complet du script analyse_croisee_json_triade.py



Ce que fait ce script (version claire et synthétique)

	Ce script analyse les clusters de similarité audio générés par ton pipeline précédent 
	(chaque cluster est un fichier JSON du type cluster_locuteur_<discussion>.json).
	Il construit une matrice multi‑niveaux permettant d’identifier trois types de relations entre discussions :
		1. Réciprocité simple (niveau 1 — bleu)

				A inclut B et B inclut A.
				→ Relation symétrique.
				
		2. Dominance / Asymétrie (niveau 2 — orange)

				A inclut B mais B n’inclut pas A.
				→ Relation directionnelle, non réciproque.
				
		3. Triade renforcée (niveau 3 — violet)

				A inclut B, B inclut A, et il existe un C tel que :
				A → C → B
				→ Relation triangulaire montrant une structure de cluster plus dense.
				
		4. Export des résultats

			Le script génère :
			🔹 Un CSV

			matrice_cooccurrence_filtree_<timestamp>.csv  
			→ matrice brute 0/1 (sans les niveaux 2 et 3).
			🔹 Une heatmap zébrée

			heatmap_croisement_triade_<timestamp>.png  
			→ couleurs distinctes pour les 4 niveaux (0,1,2,3).
			🔹 Une synthèse console

			Pour chaque discussion :

				nombre de réciprocités,
				nombre d’asymétries,
				nombre de triades.
    
    
Objectif

	Ce script analyse les clusters générés par l’analyse de similarité audio et produit une matrice multi‑niveaux 
	permettant d’identifier :

		les relations réciproques simples,
		les relations asymétriques (dominance),
		les triades renforcées (structures triangulaires),
		les discussions isolées.

	Il génère une heatmap zébrée, un CSV, et une synthèse console.

Fonctionnement

	Chargement des clusters

		Le script parcourt récursivement un dossier (resultats_analyse/) et récupère tous les fichiers :

		cluster_locuteur_<discussion>.json

		Chaque fichier contient :

			la discussion de référence,
			les membres de son cluster.
    
Construction de la matrice

	Une matrice carrée est construite :
		| Référence ↓ | Cible → | Valeur |
		| ----------- | ------- | ------ |
		| 0           | Aucun lien |     |
		| 1           | Réciproque simple |    |
		| 2           | Asymétrique / dominant |    |
		| 3           | Triade renforcée |     |

Détection des triades

			Pour chaque paire réciproque A ↔ B, le script cherche un C tel que :

			A → C → B

			Si oui → relation renforcée (niveau 3).

Heatmap zébrée

    colonnes alternées grisées pour lisibilité,
    palette personnalisée,
    légende détaillée.
    
Export

    matrice_cooccurrence_filtree_<timestamp>.csv
    heatmap_croisement_triade_<timestamp>.png
    
Structure recommandée

			project/
			│ analyser_croisement_clusters_triade.py
			│ resultats_analyse/
			│   res-discussionA/
			│      cluster_locuteur_discussionA.json
			│   res-discussionB/
			│      cluster_locuteur_discussionB.json
			│   ...
			│ resultats_croisement/
			│   matrice_cooccurrence_filtree_*.csv
			│   heatmap_croisement_triade_*.png

Exécution

	python analyser_croisement_clusters_triade.py

