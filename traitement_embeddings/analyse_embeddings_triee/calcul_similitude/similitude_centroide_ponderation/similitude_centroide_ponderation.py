import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from pathlib import Path
from itertools import product, combinations
import csv
import ast
from datetime import datetime

# script : tester_methodes1.py
# recherche pour amelioration des resultats
# approche par calcul a partir du centroide, ou par une ponderation de la durée

# ==========================================
# 0. Analyse du JSON Pyannote 
# ==========================================
def analyser_discussions_valides(chemin_json, nom_locuteur):
    """
    Objectif : Éviter que le script ne plante en essayant d'analyser des 
    discussions où le locuteur n'a pas parlé. On lit le JSON pour compter 
    les prises de parole de notre cible avant de faire le moindre calcul.
    Analyse le JSON Pyannote pour identifier les discussions où
    le locuteur cible possède au moins deux segments.
    Cela évite de lancer des calculs sur des discussions inutiles.
    """
    discussions_valides = {}
    
    if not os.path.exists(chemin_json):
        print(f"❌ ERREUR : Le fichier JSON '{chemin_json}' est introuvable.")
        return {}

    with open(chemin_json, 'r', encoding='utf-8') as f:
        donnees = json.load(f)

    for element in donnees.get('files', []):
        nom_fichier = element.get('file', '')
        id_discussion = Path(nom_fichier).stem 
        
        compte_segments = sum(1 for turn in element.get('turns', []) if turn.get('speaker') == nom_locuteur)
        
        if compte_segments > 1:
            discussions_valides[id_discussion] = compte_segments
            
    return dict(sorted(discussions_valides.items(), key=lambda item: item[1], reverse=True))

def obtenir_durees_json(chemin_json):
    """
    Extrait les durées de chaque segment pour la pondération.
    Produit un dictionnaire : {ID_segment: durée}
    """
    durees = {}
    if not os.path.exists(chemin_json): 
        return durees
        
    with open(chemin_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for f_data in data.get('files', []):
            nom = Path(f_data['file']).stem
            compteurs = {}
            for turn in f_data.get('turns', []):
                spk = str(turn['speaker'])
                compteurs[spk] = compteurs.get(spk, 0) + 1
                # Format supposé: audio-1234_operateur_1
                utt_id = f"{nom}_{spk}_{compteurs[spk]}"
                durees[utt_id] = turn['end'] - turn['start']
    return durees

# ==========================================
# 1. Fonctions de Chargement (Aiguillage)
# ==========================================
def charger_segments_tries_pt(dossier_source, id_discussion, nom_locuteur):
    """
    Charge les fichiers .pt correspondant à une discussion et un locuteur.
    """
    embeddings = {}
    chemin = Path(dossier_source)
    
    fichiers_bruts = list(chemin.rglob('*.pt'))
    fichiers_filtres = [f for f in fichiers_bruts if id_discussion in str(f) and nom_locuteur in str(f)]
    for fichier in sorted(fichiers_filtres):
        try:
            tensor = torch.load(fichier, weights_only=True)
            if tensor.is_cuda: tensor = tensor.cpu()
            embeddings[fichier.stem] = tensor.detach().squeeze()
        except Exception as e:
            pass
    return embeddings

def charger_embeddings_csv(chemin_csv):
    """ 
    Charge un fichier CSV/TSV au format :
    Lit un fichier d'embeddings où chaque ligne suit le format : ID_segment|val1 val2 val3...
    """
    corpus = {}
    with open(chemin_csv, 'r', encoding='utf-8') as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            if '|' in ligne:
                utt_id, valeurs_str = ligne.split('|', 1)
            else:
                continue
                
            valeurs_brutes = valeurs_str.split()
            try:
                valeurs = [float(x) for x in valeurs_brutes]
            except ValueError:
                continue 
                
            if valeurs:
                corpus[utt_id] = torch.tensor(valeurs, dtype=torch.float32)
                
    return corpus

def filtrer_embeddings_memoire(corpus_en_memoire, id_discussion, nom_locuteur):
    """
    Filtre un dictionnaire d'embeddings déjà chargé en mémoire.
    """
    embeddings_filtres = {}
    for utt_id, tensor in corpus_en_memoire.items():
        if id_discussion in utt_id and nom_locuteur in utt_id:
            embeddings_filtres[utt_id] = tensor.squeeze()
            
    return dict(sorted(embeddings_filtres.items()))

# ==========================================
# 2. Fonctions de Calcul Mathématique et Agrégation
# ==========================================
def calculer_intra_discussion_agrege(embeddings, durees, methode="ponderation"):
    """
    Calcule la similarité intra-discussion en utilisant l'agrégation.
    - Calcule la similarité des deux moitiés (Pairs vs Impairs)
    - Calcule la similarité de chaque segment contre le profil global
    - comparaison segment → profil global
    - split-half (pairs vs impairs)
    """
    noms = list(embeddings.keys())
    n = len(noms)
    
    # 1. Méthode Segment vs Profil Global
    # On crée le profil parfait de cette discussion
    profil_global = agreger_profil_vocal(embeddings, durees, methode=methode)
    
    scores_uniques = []
    matrice_sim = np.zeros((n, n))
    
    for i in range(n):
        # On compare chaque segment au profil global (au lieu de le comparer aux autres segments)
        sim_segment_profil = F.cosine_similarity(embeddings[noms[i]].unsqueeze(0), profil_global.unsqueeze(0)).item()
        scores_uniques.append(sim_segment_profil)
        
        # Pour ne pas casser ton graphique (01_matrice_intra), on simule une matrice
        # où chaque ligne montre la force du segment par rapport au profil global
        for j in range(n):
            matrice_sim[i, j] = F.cosine_similarity(embeddings[noms[i]].unsqueeze(0), embeddings[noms[j]].unsqueeze(0)).item()

    moyenne_segment_vs_profil = np.mean(scores_uniques)

    # 2. Méthode Split-Half (Pairs vs Impairs)
    if n >= 2:
        emb_groupe1 = {noms[i]: embeddings[noms[i]] for i in range(0, n, 2)}
        emb_groupe2 = {noms[i]: embeddings[noms[i]] for i in range(1, n, 2)}
        
        profil1 = agreger_profil_vocal(emb_groupe1, durees, methode=methode)
        profil2 = agreger_profil_vocal(emb_groupe2, durees, methode=methode)
        
        sim_split_half = F.cosine_similarity(profil1.unsqueeze(0), profil2.unsqueeze(0)).item()
    else:
        sim_split_half = 1.0

    return matrice_sim, scores_uniques, sim_split_half, n

def calculer_inter_discussion(emb_ref, emb_cible):
    """
    Calcule la similarité inter-discussion :
    produit cartésien des embeddings.
    """
    if not emb_ref or not emb_cible:
        return [], 0.0
        
    scores = [F.cosine_similarity(ref, cible, dim=0).item() for ref, cible in product(emb_ref.values(), emb_cible.values())]
    return scores, np.mean(scores)

def agreger_profil_vocal(embeddings, durees=None, methode="centroide"):
    """
    Transforme plusieurs segments en un seul vecteur représentatif.
    - centroide : moyenne simple
    - ponderation : moyenne pondérée par la durée
    """
    if not embeddings: return None
    
    vecs = torch.stack(list(embeddings.values()))
    
    if methode == "centroide":
        return vecs.mean(dim=0)
    
    elif methode == "ponderation" and durees:
        # On utilise .get(k, 2.0) pour éviter un crash si une durée manque
        poids = torch.tensor([durees.get(k, 2.0) for k in embeddings.keys()], dtype=torch.float32)
        poids = poids / poids.sum()
        # Calcul : (Vecteurs * Poids)
        return (vecs.T * poids).sum(dim=1)
        
    return vecs.mean(dim=0)

# ==========================================
# 3. Visualisations et Exportations
# ==========================================
def generer_rapports(matrice_intra, scores_intra, moyenne_intra, resultats_inter, nom_ref, dossier_sortie="resultats_analyse"):
    """
    Génère :
    - heatmap intra-discussion
    - barres horizontales inter-discussion
    - CSV récapitulatif
    """


    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sns.set_theme(style="whitegrid")
    
    # --- A. La Matrice Analytique ---
    plt.figure(figsize=(8, 6))
    n = matrice_intra.shape[0]
    
    masque_symetrie = np.tril(np.ones_like(matrice_intra, dtype=bool), k=-1)
    limites = [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmap_paliers = mcolors.ListedColormap(sns.color_palette("viridis", n_colors=len(limites) - 1))
    norm_paliers = mcolors.BoundaryNorm(limites, cmap_paliers.N)
    
    ax = sns.heatmap(matrice_intra, annot=False, mask=masque_symetrie, cmap=cmap_paliers, norm=norm_paliers, 
                     cbar_kws={"ticks": limites}, xticklabels=range(1, n + 1), yticklabels=range(1, n + 1))
    ax.invert_yaxis() 
    plt.title(f"Matrice Intra - Réf: {nom_ref}\n(Moyenne hors diagonale: {moyenne_intra:.3f})")
    plt.yticks(rotation=0); plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{dossier_sortie}/01_matrice_intra_{timestamp}.png", dpi=300)
    plt.close()

    # --- B. Le Bâton Gradué Horizontal ---
    plt.figure(figsize=(10, max(6, len(resultats_inter) * 0.8))) 
    cibles_triees = sorted(list(resultats_inter.items()), key=lambda x: x[1]["moyenne"])
    
    discussions = [x[0] for x in cibles_triees] + [f"{nom_ref} (RÉF)"]
    moyennes = [x[1]["moyenne"] for x in cibles_triees] + [moyenne_intra]
    couleurs = list(sns.color_palette("Blues", len(cibles_triees))) + ["#2ecc71"]
    
    barres = plt.barh(discussions, moyennes, color=couleurs, edgecolor='black', height=0.6)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.axvline(x=moyenne_intra, color='#e74c3c', linestyle='--', linewidth=2, label=f"Réf ({moyenne_intra:.3f})")
    
    for barre in barres:
        largeur = barre.get_width()
        x_offset = largeur + 0.02 if largeur >= 0 else largeur - 0.02
        plt.text(x_offset, barre.get_y() + barre.get_height() / 2, f"{largeur:.3f}", 
                 ha='left' if largeur >= 0 else 'right', va='center', color='black', fontweight='bold', fontsize=10)

    plt.title("Moyenne de Similitude par Discussion\n(Échelle -1 à 1)", fontsize=14, fontweight='bold')
    plt.xlim(-1.0, 1.15) 
    plt.legend(loc="lower right") 
    plt.tight_layout()
    plt.savefig(f"{dossier_sortie}/02_moyennes_horizontales_{timestamp}.png", dpi=300)
    plt.close()

    # --- C. Export CSV ---
    fichier_csv = f"{dossier_sortie}/rapport_similitudes_{timestamp}.csv"
    with open(fichier_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Type", "ID Discussion", "Nb Segments", "Similitude Moyenne"])
        writer.writerow(["INTRA (Réf)", nom_ref, n, round(moyenne_intra, 4)])
        for id_cible, data in reversed(cibles_triees):
            writer.writerow(["INTER", id_cible, data["nb_segments"], round(data["moyenne"], 4)])


# ==========================================
# Exécution Principale
# ==========================================
if __name__ == "__main__":
    """
    Pipeline complet :
    1. Analyse du JSON
    2. Chargement des embeddings
    3. Agrégation + calculs
    4. Visualisations + export
    """ 

    # --- CONFIGURATION MANUELLE ---
    SOURCE_DONNEES = "data/embeddings_nettoyes.csv" 
    FICHIER_JSON_PYANNOTE = "segmentation_audios_nettoyes.json"
    LOCUTEUR_CIBLE = "operateur"

    print("--- ETAPE 1 : Analyse des discussions ---")
    discussions_valides = analyser_discussions_valides(FICHIER_JSON_PYANNOTE, LOCUTEUR_CIBLE)
    
    if not discussions_valides:
        print("Aucune discussion valide trouvée pour ce locuteur. Arrêt.")
        exit()

    print(f"\n✅ {len(discussions_valides)} discussions contenant l'opérateur ont été trouvées :")
    liste_ids = list(discussions_valides.keys())
    
    for i, id_disc in enumerate(liste_ids):
        print(f"  [{i+1}] {id_disc} ({discussions_valides[id_disc]} segments)")

    choix = input(f"\n👉 Entrez le numéro de la discussion à utiliser comme RÉFÉRENCE (1-{len(liste_ids)}) : ")
    
    try:
        index_choix = int(choix) - 1
        DISCUSSION_REFERENCE = liste_ids[index_choix]
    except (ValueError, IndexError):
        print("❌ Choix invalide. Le script va utiliser la première discussion par défaut.")
        DISCUSSION_REFERENCE = liste_ids[0]

    DISCUSSIONS_A_COMPARER = [d for d in liste_ids if d != DISCUSSION_REFERENCE]

    print(f"\n⭐ Référence choisie : {DISCUSSION_REFERENCE}")
    print(f"🔄 Comparaison prévue avec {len(DISCUSSIONS_A_COMPARER)} autre(s) discussion(s).")

    # ----------------------------------------------------
    # ETAPE 2 : Aiguillage (PT, PKL, ou CSV)
    # ----------------------------------------------------
    est_format_pkl = str(SOURCE_DONNEES).endswith('.pkl')
    est_format_csv = str(SOURCE_DONNEES).endswith('.csv') or str(SOURCE_DONNEES).endswith('.tsv')
    
    corpus_en_memoire = None
    
    if est_format_pkl:
        print(f"\n📦 Chargement du fichier Kiwano PKL...")
        try:
            from kiwano.embedding import load_embeddings
            corpus_en_memoire = load_embeddings(SOURCE_DONNEES)
        except ImportError:
            corpus_en_memoire = torch.load(SOURCE_DONNEES, weights_only=False)
            
    elif est_format_csv:
        print(f"\n📄 Chargement du fichier CSV/TSV...")
        corpus_en_memoire = charger_embeddings_csv(SOURCE_DONNEES)
        print(f" -> {len(corpus_en_memoire)} segments chargés en mémoire.")
        
    else:
        print(f"\n📁 Lecture des dossiers .pt...")

    def extraire(id_disc):
        if est_format_pkl or est_format_csv:
            return filtrer_embeddings_memoire(corpus_en_memoire, id_disc, LOCUTEUR_CIBLE)
        else:
            return charger_segments_tries_pt(SOURCE_DONNEES, id_disc, LOCUTEUR_CIBLE)

    # ----------------------------------------------------
    # ETAPE 3 : Calculs avec Agrégation (Centroïde ou Pondération)
    # ----------------------------------------------------
    # ⚠️ Il faut définir la méthode ici, tout au début de l'étape 3
    METHODE = "ponderation" # Options : "ponderation" ou "centroide"
    
    print(f"\n📊 Analyse de la référence (Intra-discussion)...")
    emb_ref_brut = extraire(DISCUSSION_REFERENCE)
    
    if len(emb_ref_brut) < 2:
        print("❌ Erreur : Pas assez de segments trouvés dans les embeddings.")
    else:
        # Récupération des durées (nécessaire pour la pondération)
        durees_dict = obtenir_durees_json(FICHIER_JSON_PYANNOTE)

        # 1. Calcul intra-discussion avec agrégation (Split-Half)
        matrice_intra, scores_intra, moyenne_intra, nb_seg_ref = calculer_intra_discussion_agrege(
            emb_ref_brut, 
            durees=durees_dict, 
            methode=METHODE
        )
        print(f" -> {nb_seg_ref} segments traités. Moyenne intra (Split-Half) : {moyenne_intra:.4f}")

        print(f"\n📊 Analyse avec la méthode : {METHODE.upper()}")

        # AGRÉGATION : On transforme les X segments en 1 seul vecteur "Profil"
        profil_ref = agreger_profil_vocal(emb_ref_brut, durees=durees_dict, methode=METHODE)

        # 2. Comparaison Inter-discussion
        resultats_inter = {}
        for id_cible in DISCUSSIONS_A_COMPARER:
            emb_cible_brut = extraire(id_cible)
            
            if emb_cible_brut:
                # Profilage de la discussion cible
                profil_cible = agreger_profil_vocal(emb_cible_brut, durees=durees_dict, methode=METHODE)
                
                # Similarité cosinus entre les 2 vecteurs agrégés (1D)
                sim = F.cosine_similarity(profil_ref.unsqueeze(0), profil_cible.unsqueeze(0)).item()
                
                resultats_inter[id_cible] = {"moyenne": sim, "nb_segments": len(emb_cible_brut)}
                print(f" -> Comparaison {id_cible} ({len(emb_cible_brut)} segs) : {sim:.4f}")
        
        # Génération des résultats et graphiques
        generer_rapports(matrice_intra, scores_intra, moyenne_intra, resultats_inter, DISCUSSION_REFERENCE)
        print("\n🎉 Terminé ! Consultez le dossier 'resultats_analyse'.")
