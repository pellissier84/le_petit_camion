

#  README — Script de Listing des Fichiers Audio Pyannote

##  Objectif

Ce script extrait automatiquement la **liste des fichiers audio** présents dans un JSON généré par un système de diarisation (ex. *pyannote.audio*).  
Il vérifie la validité du JSON, affiche les noms des fichiers trouvés et les sauvegarde dans un fichier texte :

```
liste_fichiers_nexsis.txt
```

---

##  Entrée attendue
Le script lit un fichier JSON structuré avec une clé `"files"` contenant des objets du type :

```json
{
  "files": [
    {"file": "audio1.wav"},
    {"file": "audio2.wav"}
  ]
}
```


##  Utilisation
1. Place ton fichier JSON (ex. `segmentation_audios_Nexsis.json`) dans le même dossier que le script.  
2. Exécute simplement :

```bash
python lister_fichiers.py
```

3. Le script :
   - vérifie l’existence du JSON  
   - valide sa structure  
   - extrait les noms des fichiers audio  
   - les affiche dans la console  
   - les écrit dans `liste_fichiers_nexsis.txt`

---

## 📦 Résultat
Un fichier texte contenant **un nom de fichier audio par ligne**, prêt à être utilisé dans d’autres scripts :

```
audio1.wav
audio2.wav
audio3.wav
```

Tu peux ensuite l’utiliser pour un traitement audio, par exemple :  
- découpage locuteurs  
- vérification fichiers
- depuis la liste, lancer le script python decoupage_audio_liste.py

---

##  Compatibilité
Le script nécessite **Python 3.x**.  
Python 2.x provoque des erreurs d’encodage UTF‑8.  
Plus d’infos : python2 vs python3

---


