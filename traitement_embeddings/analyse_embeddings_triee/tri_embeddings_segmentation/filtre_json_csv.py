import json
import os

# ==========================================
# Configuration
# ==========================================
FICHIER_JSON = "segmentation_audios_Nexsis.json"
FICHIER_CSV = "embeddings.csv"

SORTIE_JSON = "segmentation_audios_nettoyes.json"
SORTIE_CSV = "embeddings_nettoyes.csv"

# Mapping pour normaliser les noms de locuteurs
# (permet d’éviter des incohérences dans les IDs)
MAPPING_LOCUTEURS = {
    "operateur": "operateur",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4"
}

def nettoyer_donnees():
    """
    Fonction principale :
    - Nettoie le JSON en supprimant les segments trop courts
    - Reconstruit les IDs des segments valides
    - Filtre le CSV pour ne conserver que les embeddings correspondants
    """

    # Ensemble des IDs valides (servira au filtrage du CSV)
    ids_valides = set()
    # Structure du JSON nettoyé
    donnees_propres = {"files": []}
    
    print(f"⏳ Lecture et filtrage de {FICHIER_JSON}...")

    # Chargement du JSON brut
    with open(FICHIER_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Parcours des fichiers audio
    for file_data in data.get('files', []):
        nom_base = file_data['file'].replace('.wav', '') # nom du fichier sans extension
        compteurs = {}  # compteur par locuteur pour reconstruire les IDs
        turns_valides = []  # liste des segments conservés

        # Parcours des segments ("turns")
        for turn in file_data.get('turns', []):
            duree = turn['end'] - turn['start']
            
            # Filtrage : on ne garde que les segments > 2 secondes
            if duree > 2.0:
                speaker_raw = str(turn['speaker'])
                speaker_name = MAPPING_LOCUTEURS.get(speaker_raw, speaker_raw)
                
                # Mise à jour du compteur
                key = (nom_base, speaker_name)
                compteurs[key] = compteurs.get(key, 0) + 1
                
                # Reconstitution de l'ID pour le filtrage du CSV
                valide_id = f"{nom_base}_{speaker_name}_{compteurs[key]}"
                ids_valides.add(valide_id)
                
                # Ajout du segment filtré
                turns_valides.append(turn)
        
        # Si la discussion a des segments valides, on l'ajoute au JSON nettoyé
        if turns_valides:
            file_data['turns'] = turns_valides
            donnees_propres['files'].append(file_data)

    # Sauvegarde du JSON nettoyé
    with open(SORTIE_JSON, 'w', encoding='utf-8') as f:
        json.dump(donnees_propres, f, indent=4)
    print(f"✅ {SORTIE_JSON} généré.")

    # 2. Filtrage du CSV

    print(f"⏳ Filtrage de {FICHIER_CSV}...")
    conserve = 0

        # Lecture du CSV brut + écriture du CSV filtré
    with open(FICHIER_CSV, 'r', encoding='utf-8') as f_in, \
         open(SORTIE_CSV, 'w', encoding='utf-8') as f_out:
        for ligne in f_in:
            # Format attendu : "ID | embedding1 embedding2 ..."
            if '|' in ligne:
                id_csv = ligne.split('|')[0].strip()

                 # On conserve uniquement les IDs présents dans ids_valides
                if id_csv in ids_valides:
                    f_out.write(ligne)
                    conserve += 1
    
    print(f"✅ {SORTIE_CSV} généré avec {conserve} segments conservés.")

# Point d'entrée du script

if __name__ == "__main__":

    # Vérification de la présence des fichiers sources
    if not os.path.exists(FICHIER_JSON) or not os.path.exists(FICHIER_CSV):
        print("❌ Erreur : Fichiers sources manquants.")
    else:
        nettoyer_donnees()
