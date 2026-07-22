# 🔍 Outil d'Inspection Unitaire des Embeddings (`extract_embeddings_individuel.py`)

Ce script est un utilitaire de débogage et de vérification. Contrairement au pipeline d'analyse globale, 
il permet d'isoler chirurgicalement **un seul segment audio** dans un vaste corpus pour vérifier 
que son vecteur mathématique (embedding) a été correctement généré et possède la bonne dimension (ex: 256).

## ✨ Fonctionnalités
- **Navigation hiérarchisée :** Pour éviter d'inonder le terminal avec des milliers de locuteurs, 
		le menu interactif fonctionne en deux étapes : sélection de l'appel (discussion), 
		puis sélection du segment exact au sein de cet appel.
- **Tri intelligent (Naturel) :** Les appels sont triés chronologiquement par leur numéro d'identification, 
		et les locuteurs sont triés mathématiquement (le segment 2 apparaît bien avant le segment 10).
- **Faible empreinte RAM :** Le script lit les fichiers ligne par ligne, ce qui lui permet d'inspecter 
		des fichiers d'embeddings de plusieurs gigaoctets sans saturer la mémoire de la machine.

## 🛠️ Configuration
Ouvrez le script `extract_embeddings_individuel.py` et vérifiez que les chemins pointent vers vos fichiers textuels :
- `emb_from_file` : Le fichier contenant les vecteurs (format `ID|val1 val2...`).(embeddings.csv)
- `fichier_global` : Le fichier de catalogue listant tous les segments extraits.(gobal.csv)

## 🚀 Utilisation
Lancez simplement le script dans votre terminal :
```bash
python extract_embeddings_individuel.py.py


exemple de sortie :

Recherche de la représentation pour : audio-1775031713.41765_operateur_20 ...
Sucessfully found corresponding representation for audio-1775031713.41765_operateur_20

Dimension du vecteur (Shape) : (256,)
