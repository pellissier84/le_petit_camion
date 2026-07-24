import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from pathlib import Path
from itertools import product, combinations
import csv
import ast
from datetime import datetime

# script :similitude_trois_meilleurs_auto.py
# amelioration des resultats on prend comme une reference
# les 3 meilleurs similitude cosinus par rapport au centroide totale
# meilleure moyenne intra et inter discussion
# NOUVEAU : Exécution en boucle automatique sur toutes les discussions

# ==========================================
# 0. Analyse du JSON Pyannote : identification des discussions
# ==========================================
def analyser_discussions_valides(chemin_json, nom_locuteur):
    """
    Parcourt le JSON de segmentation pour lister les discussions où le locuteur 
    cible prend la parole au moins 2 fois.
    Retourne un dictionnaire trié : { "id_discussion": nombre_de_segments }
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
        
        # Compte combien de fois le locuteur intervient dans ce fichier
        compte_segments = sum(1 for turn in element.get('turns', []) if turn.get('speaker') == nom_locuteur)
        # Seules les discussions avec matière à comparaison sont retenues
        if compte_segments > 1:
            discussions_valides[id_discussion] = compte_segments
    
    # Tri par ordre décroissant de segments pour faciliter la sélection de la référence        
    return dict(sorted(discussions_valides.items(), key=lambda item: item[1], reverse=True))

# ==========================================
# 1. Fonctions de Chargement des embeddings
# ==========================================
def charger_segments_tries_pt(dossier_source, id_discussion, nom_locuteur):
    """Charge les tenseurs depuis un dossier rempli de fichiers .pt individuels."""
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
    """Charge l'ensemble des embeddings depuis un fichier texte (CSV/TSV)."""
    corpus = {}
    with open(chemin_csv, 'r', encoding='utf-8') as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne: continue
                
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
    """Filtre rapidement le dictionnaire global en mémoire pour isoler une discussion."""
    embeddings_filtres = {}
    for utt_id, tensor in corpus_en_memoire.items():
        if id_discussion in utt_id and nom_locuteur in utt_id:
            embeddings_filtres[utt_id] = tensor.squeeze()
    return dict(sorted(embeddings_filtres.items()))

# ==========================================
# 2. Fonctions de Calcul Mathématique (Centroïdes)
# ==========================================
def obtenir_moyenne_top_k(embeddings, k=3):
    """
    Isole les 'k' meilleurs segments (les plus proches du centroïde global) 
    et calcule leur moyenne pour former le profil de référence robuste.
    """
    tensors = list(embeddings.values())
    if not tensors:
        return None, []
    
    # 1. Calcul du centroïde (la moyenne de TOUS les segments de la discussion)
    centroid = torch.mean(torch.stack(tensors), dim=0)
    
    # 2. Évaluation de la pureté : chaque segment est comparé au centroïde
    scores = {}
    for nom, emb in embeddings.items():
        scores[nom] = F.cosine_similarity(centroid, emb, dim=0).item()
        
    # 3. Sélection des K meilleurs segments (les plus proches de la moyenne parfaite)
    top_k_noms = sorted(scores, key=scores.get, reverse=True)[:min(k, len(scores))]
    top_k_tensors = [embeddings[nom] for nom in top_k_noms]
    
    # 4. Création de la nouvelle empreinte lissée
    moyenne_top_k = torch.mean(torch.stack(top_k_tensors), dim=0)
    
    return moyenne_top_k, top_k_noms

def calculer_intra_discussion(embeddings, k=3):
    """
    Calcule la similarité entre le profil moyen des top-K et tous les 
    *autres* segments de la même discussion.
    """
    k_reel = min(k, len(embeddings))
    moyenne_top_k, top_k_noms = obtenir_moyenne_top_k(embeddings, k_reel)
    
    scores_intra = []
    noms_evalues = []
    
    # On teste l'empreinte parfaite contre tous les segments exclus du Top-K
    for nom, emb in embeddings.items():
        if nom not in top_k_noms:
            sim = F.cosine_similarity(moyenne_top_k, emb, dim=0).item()
            scores_intra.append(sim)
            noms_evalues.append(nom)
            
    # Si la discussion avait exactement K segments, scores_intra sera vide.
    moyenne_globale = np.mean(scores_intra) if scores_intra else 1.0
    
    return scores_intra, moyenne_globale, top_k_noms, noms_evalues

def calculer_inter_discussion(emb_ref, emb_cible, k=3):
    """
    Compare le profil moyen des top-K de la référence avec le profil moyen
    des top-K de la discussion cible.
    """
    moyenne_top_k_ref, _ = obtenir_moyenne_top_k(emb_ref, min(k, len(emb_ref)))
    moyenne_top_k_cible, top_k_noms_cible = obtenir_moyenne_top_k(emb_cible, min(k, len(emb_cible)))
    
    if moyenne_top_k_ref is None or moyenne_top_k_cible is None:
        return [], 0.0, []
        
    score_unique = F.cosine_similarity(moyenne_top_k_ref, moyenne_top_k_cible, dim=0).item()
    
    return [score_unique], score_unique, top_k_noms_cible

# ==========================================
# 3. Visualisations et Exportations 
# ==========================================
def generer_rapports(scores_intra, moyenne_intra, noms_evalues, resultats_inter, nom_ref, top_k_noms, nom_locuteur, dossier_sortie="resultats_analyse"):
    """Génère les graphiques et le journal de bord (CSV) auditable."""
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sns.set_theme(style="whitegrid")
    
    # --- A. Vecteur Analytique intra-discussion ---
    if scores_intra: 
        plt.figure(figsize=(max(8, len(noms_evalues) * 0.8), 6.5))
        matrice_1d = np.array(scores_intra).reshape(1, -1)
        
        limites = [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cmap_paliers = mcolors.ListedColormap(sns.color_palette("viridis", n_colors=len(limites) - 1))
        norm_paliers = mcolors.BoundaryNorm(limites, cmap_paliers.N)
        
        parametres_cbar = {
            "ticks": limites,
            "orientation": "horizontal", 
            "pad": 0.8,                  
            "shrink": 0.6                
        }
        
        ax = sns.heatmap(matrice_1d, annot=True, fmt=".2f", cmap=cmap_paliers, norm=norm_paliers, 
                         cbar_kws=parametres_cbar, xticklabels=noms_evalues, yticklabels=[f"Moyenne\nTop-{len(top_k_noms)}"])
         
        plt.title(f"Intra - Réf: {nom_ref}\n(Moyenne des restants: {moyenne_intra:.3f})")
        plt.yticks(rotation=0, fontsize=10, fontweight='bold') 
        plt.xticks(rotation=45, ha='right', fontsize=8) 
        
        plt.tight_layout()
        plt.savefig(f"{dossier_sortie}/01_vecteur_intra_{nom_ref}.png", dpi=300)
        plt.close()

    # --- B. Le Bâton Gradué Horizontal inter-discussion ---
    plt.figure(figsize=(10, max(6, len(resultats_inter) * 0.8))) 
    cibles_triees = sorted(list(resultats_inter.items()), key=lambda x: x[1]["moyenne"])
    
    discussions = [x[0] for x in cibles_triees] + [f"{nom_ref} (RÉF Intra)"]
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

    plt.title("Comparaison des profils (Moyennes Top-3)\n(Échelle -1 à 1)", fontsize=14, fontweight='bold')
    plt.xlim(-1.0, 1.15) 
    plt.legend(loc="lower right") 
    plt.tight_layout()
    plt.savefig(f"{dossier_sortie}/02_moyennes_horizontales_{nom_ref}.png", dpi=300)
    plt.close()

    # --- C. Export CSV Avancé et Identification du Cluster ---
    fichier_csv = f"{dossier_sortie}/rapport_similitudes_{nom_ref}.csv"
    membres_cluster = []
    
    with open(fichier_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        
        writer.writerow([
            "Date Exécution", "Locuteur Cible", "Type Analyse", 
            "ID Discussion", "Nb Segments Total", f"ID du Profil (Top {len(top_k_noms)})",
            "Similitude Moyenne", "Score Min", "Score Max", "Ecart-Type", "Appartient au Cluster"
        ])
        
        min_intra = round(min(scores_intra), 4) if scores_intra else "N/A"
        max_intra = round(max(scores_intra), 4) if scores_intra else "N/A"
        ecart_type_intra = round(np.std(scores_intra), 4) if scores_intra else "N/A"
        
        writer.writerow([
            timestamp, nom_locuteur, "INTRA (Référence)", 
            nom_ref, len(scores_intra) + len(top_k_noms), ", ".join(top_k_noms),
            round(moyenne_intra, 4), min_intra, max_intra, ecart_type_intra, "RÉFÉRENCE"
        ])
        
        for id_cible, data in reversed(cibles_triees):
            dans_cluster = "NON"
            if data["moyenne"] > moyenne_intra:
                dans_cluster = "OUI"
                membres_cluster.append(id_cible)
                
            writer.writerow([
                timestamp, nom_locuteur, "INTER (Cible)", 
                id_cible, data["nb_segments"], data.get("noms_profil", "N/A"),
                round(data["moyenne"], 4), "N/A", "N/A", "N/A", dans_cluster
            ])

    # --- D. Export du fichier Cluster JSON ---
    fichier_cluster = f"{dossier_sortie}/cluster_locuteur_{nom_ref}.json"
    donnees_cluster = {
        "reference_utilisee": nom_ref,
        "seuil_intra_applique": round(moyenne_intra, 4),
        "nombre_membres": len(membres_cluster),
        "membres_cluster": membres_cluster
    }
    
    with open(fichier_cluster, mode='w', encoding='utf-8') as f:
        json.dump(donnees_cluster, f, indent=4)


# ==========================================
# Exécution Principale
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURATION MANUELLE ---
    SOURCE_DONNEES = "data/embeddings_nettoyes.csv" 
    FICHIER_JSON_PYANNOTE = "segmentation_audios_nettoyes.json"
    LOCUTEUR_CIBLE = "operateur"
    NB_TOP_SEGMENTS = 3 
    
    # OPTIONNEL : Remplir cette liste avec des noms précis (ex: ["disc_01", "disc_02"]) 
    # Si vide [], le script traitera TOUTES les discussions trouvées.
    CIBLES_SPECIFIQUES = []

    print("--- ETAPE 1 : Analyse des discussions ---")
    discussions_valides = analyser_discussions_valides(FICHIER_JSON_PYANNOTE, LOCUTEUR_CIBLE)
    
    if not discussions_valides:
        print("Aucune discussion valide trouvée pour ce locuteur. Arrêt.")
        exit()

    liste_ids_brute = list(discussions_valides.keys())
    
    # Filtrage selon les cibles spécifiques si renseignées
    if CIBLES_SPECIFIQUES:
        liste_ids = [d for d in liste_ids_brute if d in CIBLES_SPECIFIQUES]
        print(f"\n✅ Mode filtré activé : {len(liste_ids)} discussion(s) cible(s) sélectionnée(s).")
    else:
        liste_ids = liste_ids_brute
        print(f"\n✅ Mode global activé : {len(liste_ids)} discussions contenant l'opérateur seront traitées tour à tour.")

    if not liste_ids:
        print("Aucune cible valide à traiter après filtrage. Arrêt.")
        exit()

    # ----------------------------------------------------
    # ETAPE 2 : Chargement global des embeddings (UNE SEULE FOIS)
    # ----------------------------------------------------
    print("\n--- ETAPE 2 : Chargement des données en mémoire ---")
    est_format_pkl = str(SOURCE_DONNEES).endswith('.pkl')
    est_format_csv = str(SOURCE_DONNEES).endswith('.csv') or str(SOURCE_DONNEES).endswith('.tsv')
    
    corpus_en_memoire = None
    
    if est_format_pkl:
        print(f"📦 Chargement unique du fichier Kiwano PKL...")
        try:
            from kiwano.embedding import load_embeddings
            corpus_en_memoire = load_embeddings(SOURCE_DONNEES)
        except ImportError:
            corpus_en_memoire = torch.load(SOURCE_DONNEES, weights_only=False)
            
    elif est_format_csv:
        print(f"📄 Chargement unique du fichier CSV/TSV...")
        corpus_en_memoire = charger_embeddings_csv(SOURCE_DONNEES)
        print(f" -> {len(corpus_en_memoire)} segments chargés globalement en mémoire.")
        
    else:
        print(f"📁 Lecture dynamique des dossiers .pt (chargement à la volée conservé).")

    def extraire(id_disc):
        if est_format_pkl or est_format_csv:
            return filtrer_embeddings_memoire(corpus_en_memoire, id_disc, LOCUTEUR_CIBLE)
        else:
            return charger_segments_tries_pt(SOURCE_DONNEES, id_disc, LOCUTEUR_CIBLE)


    # ----------------------------------------------------
    # ETAPE 3 : Lancement de la boucle automatique
    # ----------------------------------------------------
    print("\n--- ETAPE 3 : Démarrage du traitement par lots ---")
    
    for index, DISCUSSION_REFERENCE in enumerate(liste_ids):
        print(f"\n{'='*60}")
        print(f"🔄 TRAITEMENT {index + 1}/{len(liste_ids)} | RÉFÉRENCE ACTUELLE : {DISCUSSION_REFERENCE}")
        print(f"{'='*60}")
        
        # On compare la référence actuelle avec toutes les autres
        DISCUSSIONS_A_COMPARER = [d for d in liste_ids_brute if d != DISCUSSION_REFERENCE]
        
        print(f"📊 Extraction et calculs Intra-discussion...")
        emb_ref = extraire(DISCUSSION_REFERENCE)
        
        if len(emb_ref) < 2:
            print(f"⚠️ Pas assez de segments pour {DISCUSSION_REFERENCE}. Passage au suivant.")
            continue
            
        scores_intra, moyenne_intra, top_k_noms, noms_evalues = calculer_intra_discussion(emb_ref, k=NB_TOP_SEGMENTS)
        print(f" -> Moyenne intra-discussion calculée : {moyenne_intra:.4f}")

        resultats_inter = {}
        print(f"📊 Calculs Inter-discussions ({len(DISCUSSIONS_A_COMPARER)} cibles)...")
        
        for id_cible in DISCUSSIONS_A_COMPARER:
            emb_cible = extraire(id_cible)
            if emb_cible:
                scores_inter, moyenne_inter, top_k_noms_cible = calculer_inter_discussion(emb_ref, emb_cible, k=NB_TOP_SEGMENTS)
                
                resultats_inter[id_cible] = {
                    "moyenne": moyenne_inter, 
                    "nb_segments": len(emb_cible),
                    "noms_profil": ", ".join(top_k_noms_cible)
                }

        # --- Génération des rapports ---
        dossier_dynamique = f"resultats_analyse/res-{DISCUSSION_REFERENCE}"
        generer_rapports(
            scores_intra, 
            moyenne_intra, 
            noms_evalues, 
            resultats_inter, 
            DISCUSSION_REFERENCE, 
            top_k_noms, 
            LOCUTEUR_CIBLE,
            dossier_sortie=dossier_dynamique 
        )
        
        nb_cluster = sum(1 for data in resultats_inter.values() if data["moyenne"] > moyenne_intra)
        print(f"✅ Analyse terminée : {nb_cluster} cibles dans le cluster de {DISCUSSION_REFERENCE}.")
        print(f"📁 Dossier exporté : {dossier_dynamique}")

    print("\n" + "*"*60)
    print("🎉 TRAITEMENT AUTOMATIQUE DE TOUTES LES DISCUSSIONS TERMINÉ !")
    print("*"*60 + "\n")
