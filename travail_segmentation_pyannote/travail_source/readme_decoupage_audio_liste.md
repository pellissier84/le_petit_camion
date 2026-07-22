
---

#  README — Script de Découpage Audio par Locuteur

Ce projet contient un script Python permettant de **découper automatiquement des 
fichiers audio WAV en segments individuels**, classés **par locuteur** et **par ordre d’intervention**, 
à partir d’un fichier JSON issu d’un système de diarisation (ex. *pyannote.audio*).

Le script génère pour chaque locuteur des fichiers WAV distincts, nommés selon le format :

```
nomAudio_locuteur_numero.wav
```

---

##  Objectif du script

Le script :

- lit un fichier JSON contenant les segments annotés (turns) pour chaque fichier audio ;
- charge le fichier WAV correspondant ;
- extrait chaque segment selon son **locuteur**, son **début** et sa **fin** ;
- exporte chaque segment dans un dossier dédié (`locuteurs_extraits/` par défaut) ;
- numérote automatiquement les segments pour chaque locuteur.

---

##  Structure attendue des fichiers

### 1. Fichier JSON (ex. `segmentation_audios_Nexsis.json`)

Le JSON doit contenir une liste de fichiers, chacun avec :

- `file` : nom du fichier audio annoté  
- `turns` : liste des segments, chaque segment contenant :
  - `speaker` : identifiant du locuteur  
  - `start` : début du segment (en secondes)  
  - `end` : fin du segment (en secondes)

Exemple minimal :

```json
{
  "files": [
    {
      "file": "audio1.wav",
      "turns": [
        {"speaker": "S1", "start": 0.5, "end": 2.3},
        {"speaker": "S2", "start": 3.0, "end": 5.1}
      ]
    }
  ]
}
```

---

### 2. Fichier liste (`liste_fichiers_nexsis.txt`)

Ce fichier doit contenir **un nom de fichier WAV par ligne**, par exemple :

```
audio1.wav
audio2.wav
audio3.wav
```

---

##  Installation

### Dépendances Python

Le script utilise :

- `pydub` pour manipuler les fichiers audio  
- `ffmpeg` (obligatoire pour lire/exporter les WAV)

Installation :

```bash
pip install pydub
sudo apt install ffmpeg
```

---

##  Utilisation

### Commande

Place le script dans le même dossier que :

- `segmentation_audios_Nexsis.json`
- `liste_fichiers_nexsis.txt`
- les fichiers WAV à traiter

Puis exécute :

```bash
python decoupage_audio_liste.py
```

---

## 📦 Résultat

Un dossier `locuteurs_extraits/` est créé automatiquement, contenant des fichiers nommés comme :

```
audio1_S1_1.wav
audio1_S1_2.wav
audio1_S2_1.wav
audio2_S3_1.wav
```

Chaque fichier correspond à **un segment audio d’un locuteur**.

---

##  Fonction principale

La fonction clé du script est :

### `extraire_wav_par_locuteur(chemin_json, chemin_wav, nom_etiquette_json, dossier_sortie="locuteurs_extraits")`

Elle :

- charge les segments du JSON correspondant au fichier audio ;
- découpe le WAV selon les timestamps ;
- exporte les segments dans un dossier dédié.


---

##  Exemple de log lors de l’exécution

```
Début du traitement de 12 fichiers...
Traitement de : audio1.wav
  ✓ Traitement terminé pour audio1.wav
Traitement de : audio2.wav
  ✓ Traitement terminé pour audio2.wav

Tous les fichiers ont été traités !
```




##  Gestion des erreurs

Le script gère :

- JSON introuvable  
- WAV illisible  
- absence de segments pour un fichier  
- création automatique du dossier de sortie  



##  Améliorations possibles

- Génération d’un rapport CSV des segments extraits  
- Fusion des segments par locuteur  
- Normalisation audio automatique  
- Détection des silences avant export

