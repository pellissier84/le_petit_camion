import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import csv

# Configuration stricte pour Linux (sans interface graphique)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import networkx as nx
from pathlib import Path
from datetime import datetime

# script : graphe_similitude.py

# ==========================================
# 0. Fonctions d'analyse et de chargement
# ==========================================
def analyser_discussions_valides(chemin_json, nom_locuteur):
    """
    Analyse le fichier JSON de segmentation pyannote.
    Objectif : repérer les discussions où le locuteur cible
    intervient au moins deux fois (sinon pas assez de matière).
    Retour : dictionnaire trié {id_discussion: nb_segments}
    """
    discussions_valides = {}
    if not os.path.exists(chemin_json):
        print(f"❌ ERREUR : Le fichier JSON '{chemin_json}' est introuvable.")
        return {}

    # Chargement du JSON
    with open(chemin_json, 'r', encoding='utf-8') as f:
        donnees = json.load(f)

    # Parcours des fichiers audio
    for element in donnees.get('files', []):
        nom_fichier = element.get('file', '')
        id_discussion = Path(nom_fichier).stem   # nom sans extension

        # Compte combien de fois le locuteur intervient dans ce fichier
        compte_segments = sum(1 for turn in element.get('turns', []) if turn.get('speaker') == nom_locuteur)
        # Seules les discussions avec matière à comparaison sont retenues
        if compte_segments > 1:
            discussions_valides[id_discussion] = compte_segments
    # Tri par ordre décroissant de segments pour faciliter la sélection de la référence 
    return dict(sorted(discussions_valides.items(), key=lambda item: item[1], reverse=True))


# 1. Fonctions de Chargement des embeddings

def charger_embeddings_csv(chemin_csv):
    """
    Charge un fichier CSV/TSV contenant des embeddings.
    Format attendu : utt_id | v1 v2 v3 ... vN
    Retour : dict {utt_id: tensor}
    """
    corpus = {}
    with open(chemin_csv, 'r', encoding='utf-8') as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or '|' not in ligne: continue
            utt_id, valeurs_str = ligne.split('|', 1)
            try:
                valeurs = [float(x) for x in valeurs_str.split()]
            except ValueError:
                continue  # ignore les lignes corrompues

            if valeurs:
                corpus[utt_id] = torch.tensor(valeurs, dtype=torch.float32)
    return corpus

def filtrer_embeddings_memoire(corpus_en_memoire, id_discussion, nom_locuteur):
    """
    Filtre les embeddings en mémoire pour ne garder que ceux
    appartenant à une discussion donnée + locuteur cible.
    """
    embeddings_filtres = {}

    
    for utt_id, tensor in corpus_en_memoire.items():
        # Exemple d'utt_id : "audio123_operateur_0001"
        if id_discussion in utt_id and nom_locuteur in utt_id:
            embeddings_filtres[utt_id] = tensor.squeeze()
    # Tri pour stabilité
    return dict(sorted(embeddings_filtres.items()))

# 2. Fonctions de Calcul Mathématique (Centroïdes)
def obtenir_moyenne_top_k(embeddings, k=3):
    """
    Calcule un centroïde robuste :
    1. Moyenne globale de tous les segments
    2. Score de similarité de chaque segment au centroïde
    3. Sélection des K meilleurs segments
    4. Moyenne finale = profil robuste
    """
    tensors = list(embeddings.values())
    if not tensors: return None
     # 1. Calcul du centroïde (la moyenne de TOUS les segments de la discussion)
    centroid = torch.mean(torch.stack(tensors), dim=0)
    # 2. Évaluation de la pureté : chaque segment est comparé au centroïde
    scores = {nom: F.cosine_similarity(centroid, emb, dim=0).item() for nom, emb in embeddings.items()}
     # 3. Sélection des K meilleurs segments (les plus proches de la moyenne parfaite)
    top_k_noms = sorted(scores, key=scores.get, reverse=True)[:min(k, len(scores))]
    top_k_tensors = [embeddings[nom] for nom in top_k_noms]
    return torch.mean(torch.stack(top_k_tensors), dim=0)

def calculer_inter_discussion(emb_ref, emb_cible, k=3):
    """
    Compare deux discussions :
    - calcule leur centroïde robuste (top-K)
    - renvoie la similarité cosinus
    """
    moyenne_top_k_ref = obtenir_moyenne_top_k(emb_ref, min(k, len(emb_ref)))
    moyenne_top_k_cible = obtenir_moyenne_top_k(emb_cible, min(k, len(emb_cible)))

    if moyenne_top_k_ref is None or moyenne_top_k_cible is None:
        return 0.0
    return F.cosine_similarity(moyenne_top_k_ref, moyenne_top_k_cible, dim=0).item()


# ==========================================
# Exécution Principale : Réseau Global Top 10
# ==========================================
if __name__ == "__main__":

    # --- CONFIGURATION ---
    SOURCE_DONNEES = "data/embeddings_nettoyes.csv" 
    FICHIER_JSON_PYANNOTE = "segmentation_audios_nettoyes.json"
    LOCUTEUR_CIBLE = "operateur"
    NB_TOP_SEGMENTS = 3 
    LIMITE_TOP_CIBLES = 10 # On ne garde que les 10 meilleurs audios pour chaque référence
    DOSSIER_SORTIE = "resultats_reseau_global"

    Path(DOSSIER_SORTIE).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print("--- 1. Analyse des discussions ---")
    discussions_valides = analyser_discussions_valides(FICHIER_JSON_PYANNOTE, LOCUTEUR_CIBLE)
    liste_ids = list(discussions_valides.keys())
    
    if not liste_ids:
        print("Aucune discussion valide. Arrêt.")
        exit()

    print(f" -> {len(liste_ids)} discussions trouvées. Chargement en mémoire...")
    corpus_en_memoire = charger_embeddings_csv(SOURCE_DONNEES)

    # Fonction utilitaire
    def extraire(id_disc):
        return filtrer_embeddings_memoire(corpus_en_memoire, id_disc, LOCUTEUR_CIBLE)

    # --- 2. Calcul Croisé de tout le corpus ---
    print(f"\n--- 2. Calcul du Top {LIMITE_TOP_CIBLES} croisé sur l'ensemble du corpus ---")
    
    # Dictionnaire qui stockera { "Audio_A": {"Audio_B": score, "Audio_C": score...} }
    reseau_top10 = {}

    for i, ref_actuelle in enumerate(liste_ids):
        print(f"Progression : [{i+1}/{len(liste_ids)}] Évaluation de {ref_actuelle}...")
        emb_ref = extraire(ref_actuelle)
        if len(emb_ref) < 2: continue
        
        scores_cibles = {}
        for id_cible in liste_ids:
            if id_cible != ref_actuelle:
                emb_cible = extraire(id_cible)
                if emb_cible:
                    score = calculer_inter_discussion(emb_ref, emb_cible, k=NB_TOP_SEGMENTS)
                    scores_cibles[id_cible] = score
                    
        # On ne conserve que les 10 meilleurs
        top_10 = dict(sorted(scores_cibles.items(), key=lambda item: item[1], reverse=True)[:LIMITE_TOP_CIBLES])
        reseau_top10[ref_actuelle] = top_10

    # --- 3. Construction du Graphe et Export CSV ---
    print("\n--- 3. Construction de la carte topologique ---")
    
    G = nx.DiGraph() # Graphe dirigé (avec des flèches)
    chemin_csv = f"{DOSSIER_SORTIE}/liens_top10_reseau_{timestamp}.csv"
    
    # Ajout des noeuds
    for noeud in liste_ids:
        G.add_node(noeud)

    with open(chemin_csv, mode='w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=';')
        csv_writer.writerow(["Audio Référence (Générateur)", "Audio Cible (Capturé)", "Classement", "Score Similitude", "Lien Réciproque (Noyau Dur)"])
        
        for ref, cibles in reseau_top10.items():
            classement = 1
            for cible, score in cibles.items():
                
                # Vérification de la réciprocité (Noyau Dur)
                est_reciproque = "NON"
                if cible in reseau_top10 and ref in reseau_top10[cible]:
                    est_reciproque = "OUI"
                
                # On écrit dans le CSV
                csv_writer.writerow([ref, cible, classement, round(score, 4), est_reciproque])
                classement += 1
                
                # On ajoute le lien dans le graphe
                # On ajoute l'attribut 'reciproque' pour gérer la couleur du trait plus tard
                G.add_edge(ref, cible, weight=score, reciproque=(est_reciproque=="OUI"))

    print(f"✅ Fichier CSV des connexions généré : {chemin_csv}")

    # --- 4. Dessin du Graphe Spatial ---
    plt.figure(figsize=(24, 16)) # Très grande image pour que les 70 audios respirent
    
    # Algorithme de force (Spring layout) pour regrouper les clusters naturellement
    # k contrôle la distance optimale entre les noeuds. On l'augmente pour écarter les données.
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)
    
    # Calcul de la taille des points (Plus un audio est ciblé, plus il est gros)
    degres_entrants = dict(G.in_degree())
    tailles_noeuds = [300 + (degres_entrants[n] * 100) for n in G.nodes()]
    
    # Séparation des flèches simples (unilatérales) et réciproques (Noyau dur)
    edges_simples = [(u, v) for u, v, d in G.edges(data=True) if not d['reciproque']]
    edges_reciproques = [(u, v) for u, v, d in G.edges(data=True) if d['reciproque']]

    # Dessin des points
    nx.draw_networkx_nodes(G, pos, node_size=tailles_noeuds, node_color="#3498db", edgecolors="white", linewidths=2, alpha=0.9)
    
    # ==========================================
    #  Dessin des flèches unilatérales visibles
    # ==========================================
    nx.draw_networkx_edges(G, pos, edgelist=edges_simples, edge_color="dimgray", arrows=True, arrowsize=10, width=0.8, alpha=0.6, connectionstyle="arc3,rad=0.1")
    
    # Dessin des flèches réciproques / Noyaux Durs (épaisses et rouges)
    nx.draw_networkx_edges(G, pos, edgelist=edges_reciproques, edge_color="#e74c3c", arrows=True, arrowsize=15, width=2.5, alpha=0.9)

    # Ajout des étiquettes (noms des audios)
    nx.draw_networkx_labels(G, pos, font_size=6, font_family="sans-serif", font_weight="bold")

    # Finalisation visuelle
    plt.title(f"Topologie du Corpus : Cartographie des {LIMITE_TOP_CIBLES} meilleures similitudes", fontsize=20, fontweight='bold', pad=20)
    plt.axis("off") 
    
    # ==========================================
    #  la couleur dans la légende
    # ==========================================
    patch_noeud = mpatches.Patch(color='#3498db', label='Audio (Taille = nb de fois capturé)')
    patch_simple = mpatches.Patch(color='dimgray', label='Lien Simple (Top 10 asymétrique)')
    patch_double = mpatches.Patch(color='#e74c3c', label='Noyau Dur (Top 10 réciproque entre les deux)')
    plt.legend(handles=[patch_noeud, patch_simple, patch_double], loc='upper left', fontsize=12, frameon=True)

    plt.tight_layout()
    chemin_img = f"{DOSSIER_SORTIE}/carte_topologique_{timestamp}.png"
    plt.savefig(chemin_img, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Carte topologique réseau générée : {chemin_img}")
    print("\n" + "="*50)
    print("🎉 ANALYSE DE RÉSEAU TERMINÉE !")
    print("="*50)
