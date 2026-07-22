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
from datetime import datetime

# ============================================================
# SCRIPT : similitude_duree_wav.py
# Objectif : améliorer la qualité des profils vocaux en utilisant
#            les segments les plus longs (durée > 2 sec)
#            plutôt que les segments les plus proches du centroïde.


# ==========================================
# 0. Analyse du JSON Pyannote : Identification et extraction des durées
# ==========================================
def analyser_discussions_et_durees(chemin_json, nom_locuteur):
    """
    Analyse le fichier JSON produit par Pyannote.
    - Identifie les discussions où le locuteur apparaît au moins 2 fois.
    - Calcule la durée exacte de chaque segment.
    - Conserve la numérotation originale des segments (1, 2, 3…).
    Retourne :
        discussions_valides : {id_discussion : nb_segments}
        durees_segments : {segment_id : durée_en_secondes}
    """
    discussions_valides = {}
    durees_segments = {}
    
    if not os.path.exists(chemin_json):
        print(f"❌ ERREUR : Le fichier JSON '{chemin_json}' est introuvable.")
        return {}, {}
        
    # Mapping pour uniformiser les noms des locuteurs
    MAPPING_LOCUTEURS = {
        "operateur": "operateur",
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4"
    }
    # Chargement du JSON
    with open(chemin_json, 'r', encoding='utf-8') as f:
        donnees = json.load(f)
    # Parcours des fichiers audio annotés
    for file_data in donnees.get('files', []):
        nom_fichier = file_data.get('file', '')
        nom_base = nom_fichier.replace('.wav', '')
        id_discussion = Path(nom_fichier).stem 
        
        compteurs = {}  # Compteur pour numéroter les segments
        compte_segments_cibles = 0
        
        # Parcours des segments ("turns")
        for turn in file_data.get('turns', []):
            duree = turn['end'] - turn['start']  # Durée du segment
            
            # Normalisation du nom du locuteur
            speaker_raw = str(turn.get('speaker', ''))
            speaker_name = MAPPING_LOCUTEURS.get(speaker_raw, speaker_raw)
            
            # Reprise de la logique de compteur pour conserver l'ID d'origine
            key = (nom_base, speaker_name)
            compteurs[key] = compteurs.get(key, 0) + 1
            
            # On ne garde que les segments du locuteur cible et > 2 sec
            if duree > 2.0 and speaker_name == nom_locuteur:
                valide_id = f"{nom_base}_{speaker_name}_{compteurs[key]}"
                durees_segments[valide_id] = duree
                compte_segments_cibles += 1
        
        # Une discussion est valide si elle contient au moins 2 segments du locuteur
        if compte_segments_cibles > 1:
            discussions_valides[id_discussion] = compte_segments_cibles
    
    # Tri des discussions par nombre de segments décroissant        
    return dict(sorted(discussions_valides.items(), key=lambda item: item[1], reverse=True)), durees_segments

# ==========================================
# 1. Fonctions de Chargement des embeddings
# ==========================================
def charger_segments_tries_pt(dossier_source, id_discussion, nom_locuteur):	
    """
    Charge les embeddings .pt correspondant à une discussion et un locuteur.
    """
    embeddings = {}
    chemin = Path(dossier_source)
    fichiers_bruts = list(chemin.rglob('*.pt'))
    # Filtrage par nom de discussion + locuteur
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
    Charge les embeddings depuis un fichier CSV/TSV.
    Format attendu : utt_id | val1 val2 val3 ...
    """
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
    """
    Filtre les embeddings déjà chargés en mémoire.
    """
    embeddings_filtres = {}
    for utt_id, tensor in corpus_en_memoire.items():
        if id_discussion in utt_id and nom_locuteur in utt_id:
            embeddings_filtres[utt_id] = tensor.squeeze()
    return dict(sorted(embeddings_filtres.items()))

# ==========================================
# 2. Fonctions de Calcul Temporel & Mathématique
# ==========================================
def obtenir_moyenne_plus_longs(embeddings, durees_segments, k=3):
    """
    Isole les 'k' segments les plus longs en temps 
    et calcule leur moyenne pour former le profil de référence.
    """
    # Filtrer les durées pour ne garder que celles des segments présents dans les embeddings
    durees_locales = {nom: durees_segments[nom] for nom in embeddings.keys() if nom in durees_segments}
    
    # Tri par durée décroissante et sélection des K premiers
    top_k_noms = sorted(durees_locales, key=durees_locales.get, reverse=True)[:min(k, len(durees_locales))]
    
    top_k_tensors = [embeddings[nom] for nom in top_k_noms]
    
    if not top_k_tensors:
        return None, []
        
    moyenne_top_k = torch.mean(torch.stack(top_k_tensors), dim=0)
    
    return moyenne_top_k, top_k_noms

def calculer_intra_discussion(embeddings, durees_segments, k=3):	
    """
    Compare le profil moyen des k segments les plus longs
    avec tous les autres segments de la discussion.
    """
    k_reel = min(k, len(embeddings))
    moyenne_top_k, top_k_noms = obtenir_moyenne_plus_longs(embeddings, durees_segments, k_reel)
    
    scores_intra = []
    noms_evalues = []
    
    # Calcul face aux autres segments (hors des K plus longs)
    for nom, emb in embeddings.items():
        if nom not in top_k_noms:
            sim = F.cosine_similarity(moyenne_top_k, emb, dim=0).item()
            scores_intra.append(sim)
            noms_evalues.append(nom)
            
    moyenne_globale = np.mean(scores_intra) if scores_intra else 1.0
    
    return scores_intra, moyenne_globale, top_k_noms, noms_evalues

def calculer_inter_discussion(emb_ref, emb_cible, durees_segments, k=3):	
    """
    Compare le profil des k segments les plus longs de la référence
    avec celui de la discussion cible.
    """
    moyenne_top_k_ref, _ = obtenir_moyenne_plus_longs(emb_ref, durees_segments, min(k, len(emb_ref)))
    moyenne_top_k_cible, top_k_noms_cible = obtenir_moyenne_plus_longs(emb_cible, durees_segments, min(k, len(emb_cible)))
    
    if moyenne_top_k_ref is None or moyenne_top_k_cible is None:
        return [], 0.0, []
        
    score_unique = F.cosine_similarity(moyenne_top_k_ref, moyenne_top_k_cible, dim=0).item()
    
    #  Retourne aussi les noms des segments cibles
    return [score_unique], score_unique, top_k_noms_cible

# ==========================================
# 3. Visualisations et Exportations
# ==========================================
# ==========================================
# 3. Visualisations et Exportations
# ==========================================
def generer_rapports(scores_intra, moyenne_intra, noms_evalues, resultats_inter, nom_ref, top_k_noms, durees_segments, dossier_sortie="resultats_analyse"):
    """
    Génère :
    - Heatmap intra-discussion
    - Barres horizontales inter-discussion
    - CSV complet
    """
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Le timestamp n'est plus utile ici
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    sns.set_theme(style="whitegrid")
    
    label_y = f"Moyenne\n{len(top_k_noms)} Plus Longs"

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
                         cbar_kws=parametres_cbar, xticklabels=noms_evalues, yticklabels=[label_y])
         
        plt.title(f"Intra - Réf: {nom_ref}\n(Moyenne des restants: {moyenne_intra:.3f})")
        plt.yticks(rotation=0, fontsize=10, fontweight='bold') 
        plt.xticks(rotation=45, ha='right', fontsize=8) 
        
        plt.tight_layout()
        # MODIFICATION : Utilisation de nom_ref
        plt.savefig(f"{dossier_sortie}/01_vecteur_intra_{nom_ref}.png", dpi=300)
        plt.close()

    # --- B. Le Bâton Gradué Horizontal inter-discussion ---
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

    plt.title(f"Comparaison des profils (Moyennes des {len(top_k_noms)} Plus Longs)\n(Échelle -1 à 1)", fontsize=14, fontweight='bold')
    plt.xlim(-1.0, 1.15) 
    plt.legend(loc="lower right") 
    plt.tight_layout()
    # MODIFICATION : Utilisation de nom_ref
    plt.savefig(f"{dossier_sortie}/02_moyennes_horizontales_{nom_ref}.png", dpi=300)
    plt.close()

    # --- C. Export CSV Enrichi ---
    # MODIFICATION : Utilisation de nom_ref
    fichier_csv = f"{dossier_sortie}/rapport_similitudes_{nom_ref}.csv"
    
    # Calcul de la durée du profil de référence
    duree_ref = sum([durees_segments[nom] for nom in top_k_noms])
    
    with open(fichier_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        
        # En-tête des colonnes
        writer.writerow([
            "Type Analyse", 
            "ID Discussion", 
            "Score Similitude", 
            "Segments Evalués (Total)", 
            f"ID du Profil (Top {len(top_k_noms)})", 
            "Durée Profil (sec)",
            "Score Min (Intra)",
            "Score Max (Intra)"
        ])
        
        # Ligne de la référence (INTRA)
        score_min = round(min(scores_intra), 4) if scores_intra else "N/A"
        score_max = round(max(scores_intra), 4) if scores_intra else "N/A"
        
        writer.writerow([
            "INTRA (Référence)", 
            nom_ref, 
            round(moyenne_intra, 4), 
            len(scores_intra) + len(top_k_noms), # Total des segments de la discussion
            ", ".join(top_k_noms),
            round(duree_ref, 2),
            score_min,
            score_max
        ])
        
        # Lignes des cibles (INTER)
        for id_cible, data in reversed(cibles_triees):
            writer.writerow([
                "INTER (Cible)", 
                id_cible, 
                round(data["moyenne"], 4), 
                data["nb_segments"],
                data["noms_profil"],
                round(data["duree_profil"], 2),
                "N/A", # Pas de min en inter (score unique)
                "N/A"  # Pas de max en inter (score unique)
            ])


# ==========================================
# Exécution Principale
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURATION MANUELLE ---
    SOURCE_DONNEES = "data/embeddings_nettoyes.csv" 
    FICHIER_JSON_PYANNOTE = "segmentation_audios_nettoyes.json"
    LOCUTEUR_CIBLE = "operateur"
    NB_SEGMENTS_LONGS = 3 # # Nombre de segments les plus longs à utiliser <-- Vous pouvez modifier cette valeur (ex: 2 ou 3)

    print("--- ETAPE 1 : Analyse des discussions ---")
    discussions_valides, dictionnaire_durees = analyser_discussions_et_durees(FICHIER_JSON_PYANNOTE, LOCUTEUR_CIBLE)
    
    if not discussions_valides:
        print("Aucune discussion valide trouvée pour ce locuteur.")
        exit()

    print(f"\n✅ {len(discussions_valides)} discussions contenant l'opérateur ont été trouvées :")
    # Affichage des discussions trouvées
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
    # ETAPE 3 : Calculs et Graphiques
    # ----------------------------------------------------
    print(f"\n📊 Analyse de la référence (Intra-discussion)...")
    
    # Analyse INTRA
    emb_ref = extraire(DISCUSSION_REFERENCE)
    
    if len(emb_ref) < 2:
        print("❌ Erreur : Pas assez de segments trouvés dans les embeddings.")
    else:
        # Intégration du dictionnaire des durées dans le calcul
        scores_intra, moyenne_intra, top_k_noms, noms_evalues = calculer_intra_discussion(emb_ref, dictionnaire_durees, k=NB_SEGMENTS_LONGS)
        print(f" -> {len(emb_ref)} segments traités.")
        print(f" -> {NB_SEGMENTS_LONGS} plus longs utilisés comme référence :")
        for nom in top_k_noms:
            print(f"      - {nom} ({dictionnaire_durees[nom]:.2f} sec)")
        print(f" -> Moyenne intra (hors les plus longs) : {moyenne_intra:.4f}")
        
        # Analyse INTER
        resultats_inter = {}
        print(f"\n📊 Analyse avec la référence (Inter-discussion)...")
        for id_cible in DISCUSSIONS_A_COMPARER:
            emb_cible = extraire(id_cible)
            if emb_cible:
                # MODIFICATION : On récupère top_k_noms_cible
                scores_inter, moyenne_inter, top_k_noms_cible = calculer_inter_discussion(emb_ref, emb_cible, dictionnaire_durees, k=NB_SEGMENTS_LONGS)
                
                # Calcul de la durée totale de ce profil
                duree_profil = sum([dictionnaire_durees[nom] for nom in top_k_noms_cible])
                
                resultats_inter[id_cible] = {
                    "moyenne": moyenne_inter, 
                    "nb_segments": len(emb_cible),
                    "noms_profil": ", ".join(top_k_noms_cible), # Liste lisible dans le CSV
                    "duree_profil": duree_profil
                }
                print(f" -> Comparaison {id_cible} ({len(emb_cible)} segs) : {moyenne_inter:.4f}")
        
        # --- Génération des rapports ---
        generer_rapports(scores_intra, moyenne_intra, noms_evalues, resultats_inter, DISCUSSION_REFERENCE, top_k_noms, dictionnaire_durees)
        print("\n🎉 Terminé ! Consultez le dossier 'resultats_analyse'.")
