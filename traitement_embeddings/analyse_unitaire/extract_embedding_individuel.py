import os
import numpy as np
import re

# ==========================================
# Configuration des fichiers
# ==========================================
emb_from_file = "./embeddings.csv"
fichier_global = "./global.csv"

# ==========================================
# Moteur d'Extraction
# ==========================================
def extract_representation(file_name, speaker_id):
    """From a parsed file containing speaker name (or id) and a given representation 
    (quantized units or speaker embedding), this function allows to extract the
    representation as a numpy array."""

    rep = None
    with open(file_name) as f:
        non_empty_lines = [line for line in f if line.strip()]
        for line in non_empty_lines:
            if  line.find(speaker_id) != -1: 
                rep_str = line.rsplit("|")[1]
                rep = np.fromstring(rep_str, sep=' ',  dtype=np.float64) 
                print(f"Sucessfully found corresponding representation for {speaker_id}")
                break
    if rep is None:
        print(f"Representation not found for {speaker_id}")
    return rep

# ==========================================
# Moteur de Sélection Interactive et Tri
# ==========================================
def choisir_segment_depuis_global(chemin_global):
    """
    Menu interactif en 2 étapes :
    1. Choix de la discussion (triée par numéro).
    2. Choix du segment au sein de cette discussion (trié par numéro de locuteur).
    """
    locuteur_cible = input("👉 Entrez le nom du locuteur à chercher (ex: operateur, 1, 5...) : ").strip()
    
    # Dictionnaire pour regrouper les segments par discussion : { 'audio-123': ['audio-123_op_1', ...] }
    discussions_regroupees = {}
    
    if not os.path.exists(chemin_global):
        print(f"❌ Le fichier {chemin_global} est introuvable.")
        return None

    # --- LECTURE ET REGROUPEMENT ---
    with open(chemin_global, 'r', encoding='utf-8') as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith('/'):
                continue
            
            nom_fichier = ligne.split()[0]
            
            if locuteur_cible in nom_fichier:
                id_propre = nom_fichier.replace('.wav', '')
                
                # Extraction de l'ID de la discussion (ex: audio-1775031713.41765)
                match_audio = re.search(r'(audio-[\d\.]+)', id_propre)
                if match_audio:
                    id_discussion = match_audio.group(1)
                    
                    # On ajoute le segment dans la bonne "boîte"
                    if id_discussion not in discussions_regroupees:
                        discussions_regroupees[id_discussion] = []
                    discussions_regroupees[id_discussion].append(id_propre)

    if not discussions_regroupees:
        print(f"❌ Aucun segment trouvé pour le locuteur '{locuteur_cible}'.")
        return None

    # --- ÉTAPE 1 : CHOIX DE LA DISCUSSION ---
    # Tri des discussions par leur valeur numérique (chronologique)
    liste_discussions = sorted(list(discussions_regroupees.keys()), key=lambda x: float(x.replace('audio-', '')))
    
    print(f"\n📂 {len(liste_discussions)} discussions trouvées pour '{locuteur_cible}'.")
    for i, disc in enumerate(liste_discussions):
        nb_segments = len(discussions_regroupees[disc])
        print(f"  [{i+1}] {disc} ({nb_segments} segments)")
        
    choix_disc = input(f"\n👉 Choisissez la discussion (1-{len(liste_discussions)}) : ")
    
    try:
        index_disc = int(choix_disc) - 1
        if not (0 <= index_disc < len(liste_discussions)):
            raise ValueError
    except ValueError:
        print("❌ Saisie invalide. Sélection de la première discussion par défaut.")
        index_disc = 0
        
    discussion_choisie = liste_discussions[index_disc]
    segments_de_la_discussion = discussions_regroupees[discussion_choisie]

    # --- ÉTAPE 2 : CHOIX DU SEGMENT ---
    # Fonction de tri pour classer les locuteurs (ex: operateur_2 avant operateur_10)
    def tri_locuteur(nom_segment):
        match_locuteur = re.search(r'(\d+)$', nom_segment)
        return int(match_locuteur.group(1)) if match_locuteur else 0
        
    segments_de_la_discussion.sort(key=tri_locuteur)
    
    print(f"\n✅ {len(segments_de_la_discussion)} segments dans {discussion_choisie}. Voici la liste :")
    for i, segment in enumerate(segments_de_la_discussion):
        print(f"  [{i+1}] {segment}")
        
    choix_seg = input(f"\n👉 Choisissez le segment à inspecter (1-{len(segments_de_la_discussion)}) : ")
    
    try:
        index_seg = int(choix_seg) - 1
        if 0 <= index_seg < len(segments_de_la_discussion):
            return segments_de_la_discussion[index_seg]
        else:
            print("❌ Numéro hors limite. Sélection automatique du premier segment.")
            return segments_de_la_discussion[0]
    except ValueError:
        print("❌ Saisie invalide. Sélection automatique du premier segment.")
        return segments_de_la_discussion[0]

# ==========================================
# Exécution Principale
# ==========================================
if __name__ == "__main__":
    
    print("--- OUTIL D'INSPECTION UNITAIRE DES EMBEDDINGS ---")
    utterance_id = choisir_segment_depuis_global(fichier_global)
    
    if utterance_id:
        print(f"\n🔍 Recherche de la représentation pour : {utterance_id} ...")
        
        rep = extract_representation(emb_from_file, utterance_id)
        
        if rep is not None:
            print("\n📊 Dimension du vecteur (Shape) :", rep.shape)
