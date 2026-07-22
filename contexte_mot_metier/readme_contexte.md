**README complet, structuré **, 
nom du script `evaluate_wer_metier_contexte_V2_optimise.py`.  

---

# Évaluation des mots métier avec analyse contextuelle  
Script : `evaluate_wer_metier_contexte_V2_optimise.py`

##  Objectif du script
Ce script réalise une **extraction automatique des mots métier** (lexique opérationnel SDIS/CISU) 
à partir d’un texte issu d’un appel d’urgence transcrit (WhisperX + ELAN). 
 
Il combine :

- **Regex métier** (ontologie SDIS/CISU)
- **spaCy EntityRuler** (patterns JSONL)
- **spaCy NER** (fr_core_news_lg - Modèle large pour le vocabulaire spécifique)
- **Gazetteer des communes 31** (regex générées dynamiquement)
- **Analyse contextuelle** (pondération des contextes opérationnels et gestion des faux positifs)

Il produit une liste unifiée d’entités métier détectées, calcule le taux d'erreur (WER Métier), évalue la confiance contextuelle et exporte un rapport complet.

---

##  Fonctionnalités principales

###  Extraction des mots métier
- Détection via **regex SDIS** (ontologie métier très étendue)
- Détection via **EntityRuler** (patterns CISU générés à partir des fichiers métiers)
- Détection via **spaCy NER** (LOC, ORG, PER, DATE, CARDINAL, etc.)
- Fusion intelligente des doublons

###  Normalisation du texte
- Suppression des accents
- Nettoyage des ponctuations
- Suppression des disfluences (euh, ben, ok, etc.)
- Réduction des espaces

###  Détection des communes (Haute‑Garonne)
- Chargement d’un gazetteer `.json` ou `.csv`
- Génération automatique de regex robustes :
  - accents
  - tirets
  - variantes orthographiques

###  Analyse contextuelle
Pondération des contextes opérationnels :

- Incendie urbain / véhicule / forêt / industriel  
- Accident routier  
- Détresse médicale  
- Nuisance odorante  
- Brûlage contrôlé  
- Facteurs aggravants (accident, incendie, malaise)

Chaque contexte possède :
- des **mots-clés** → score positif  
- des **mots pénalisants** → score négatif (évite les fausses détections)
- un **poids** → importance du contexte

---

##  Installation

### 1. Cloner le dépôt
```bash
git clone <votre-depot>
cd <votre-dossier>

###  Compatibilité fichiers
- JSON WhisperX (`segments`)
- EAF ELAN (via `pympi`)

---


### 2. Installer les dépendances Python
```
pip install spacy pympi-ling
```

### 3. Installer le modèle spaCy FR
```
python -m spacy download fr_core_news_sm (ou modèle lg version large)
```

### 4. Vérifier les fichiers nécessaires
- `patterns_sdis_cisu.jsonl` → patterns EntityRuler  
- `communes-haute-garonne.geojson.json` → gazetteer des communes  
- Fichiers d’entrée :
  - `.eaf`  (Annotations de référence)
  - `.json` WhisperX

---

##  Utilisation

### Commande standard
```
python evaluate_wer_metier_contexte_V2.py \
    --eaf "audio-1775031826.41778.eaf" \
    --json "audio-1775031826.41778.json"
```

### Arguments
| Argument |  Description                                       |
|----------|----------------------------------------------------|
| `--eaf`  | Fichier ELAN contenant les segments annotés        |
| `--json` | Fichier WhisperX contenant les segments transcrits |
| `--debug` *(optionnel)* | Affiche les détails de détection    |

---

##  Sortie du script
1. Rapport Console

	Le script affiche dans le terminal :

		Les statistiques globales de détection (Regex, CISU, NER).

		Le WER Métier et le pourcentage de précision.

		Les contextes majoritairement détectés.

		Une alerte sur les Faux Positifs probables.

		Le détail de chaque concept avec son score de confiance.

2. Export JSON Automatique

	Le script génère automatiquement un fichier rapport_wer_<nom_audio>.json contenant :

		Les statistiques de l'évaluation (Substitutions, Délétions, Insertions).

		L'analyse contextuelle détaillée pour intégration dans d'autres outils d'analyse.

---

##  Architecture interne (résumé)
- **Chargement spaCy**  
- **Chargement gazetteer** + génération regex  
- **Chargement EntityRuler**  
- **Ontologie métier** (regex très étendues)  
- **Contextes opérationnels** (pondération)  
- **Normalisation du texte**  
- **Extraction des entités**  
- **Fusion des doublons**  

---

##  Fichiers importants
- `evaluate_wer_metier_contexte_V2.py` → script principal  
- `patterns_sdis_cisu.jsonl` → patterns CISU  
- `communes-haute-garonne.geojson.json` → gazetteer  
- `audio-*.json` → transcription WhisperX  
- `audio-*.eaf` → segmentation ELAN  

---

##  Améliorations possibles
- Ajout d’un mode **benchmark WER métier**  
- Export des entités en **TSV** ou **JSON**  
- Visualisation des entités dans une interface Streamlit  
- Ajout d’un modèle spaCy plus puissant (`fr_core_news_md`)  

---

