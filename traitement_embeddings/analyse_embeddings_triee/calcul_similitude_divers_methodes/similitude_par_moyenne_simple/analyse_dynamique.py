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

# traitement des embeddings format .pt ou .pkl ou csv ou tsv
# choix des reference et discussions a traiter
# script : analyse_dynamique.py
# option utlisation verifier_temps_parole.py pour voir la duree des segments

# ==========================================
# 0. Analyse du JSON Pyannote : identification des discussions
# ==========================================
def analyser_discussions_valides(chemin_json, nom_locuteur):
    """
    Objectif : Éviter que le script ne plante en essayant d'analyser des 
    discussions où le locuteur n'a pas parlé. On lit le JSON pour compter 
    les prises de parole de notre cible avant de faire le moindre calcul.
    """
    discussions_valides = {}
    
    # Vérification de l'existence du fichier JSON
    if not os.path.exists(chemin_json):
        print(f"❌ ERREUR : Le fichier JSON '{chemin_json}' est introuvable.")
        return {}

    # Chargement du JSON
    with open(chemin_json, 'r', encoding='utf-8') as f:
        donnees = json.load(f)

    # Parcours des discussions
    for element in donnees.get('files', []):
        nom_fichier = element.get('file', '')
        id_discussion = Path(nom_fichier).stem 
        
        # Comptage des segments du locuteur cible
        compte_segments = sum(1 for turn in element.get('turns', []) if turn.get('speaker') == nom_locuteur)
        
        # On ne garde que les discussions avec au moins 2 segments
        if compte_segments > 1:
            discussions_valides[id_discussion] = compte_segments
     
    # Tri par nombre de segments (descendant)       
    return dict(sorted(discussions_valides.items(), key=lambda item: item[1], reverse=True))


# ==========================================
# 1. Fonctions de Chargement des embeddings (PT, PKL, CSV)
# ==========================================
def charger_segments_tries_pt(dossier_source, id_discussion, nom_locuteur):
    """
    Charge les embeddings stockés dans des fichiers .pt individuels.
    Filtre automatiquement par discussion et locuteur.
    """
    embeddings = {}
    chemin = Path(dossier_source)
    
    # Recherche de tous les fichiers .pt
    fichiers_bruts = list(chemin.rglob('*.pt'))

    # Filtrage par discussion + locuteur
    fichiers_filtres = [f for f in fichiers_bruts if id_discussion in str(f) and nom_locuteur in str(f)]

    # Chargement des tenseurs
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
    Charge un fichier CSV/TSV contenant des embeddings au format :
    ID_segment | val1 val2 val3 ...
    """
    corpus = {}
    
    # Ouverture du fichier en mode texte avec encodage universel
    with open(chemin_csv, 'r', encoding='utf-8') as f:
		# On lit le fichier ligne par ligne (très économe en RAM pour les gros fichiers)
        for ligne in f:
			# .strip() nettoie les sauts de ligne (\n) et les espaces aux extrémités
            ligne = ligne.strip()
            # Sécurité  : Ignorer les lignes totalement vides
            if not ligne:
                continue
                
            # Séparation de l'ID et de la suite des valeurs grâce au pipe "|"
            if '|' in ligne:
				# .split('|', 1) coupe la ligne en exactement 2 morceaux au premier '|' trouvé.
                # Morceau 1 = L'identifiant (ex: audio-1775..._operateur_41)
                # Morceau 2 = La longue suite de chiffres
                utt_id, valeurs_str = ligne.split('|', 1)
            else:
                continue # Ignore les lignes mal formatées ou les en-têtes
                
            # Les valeurs numériques sont séparées par des espaces
            valeurs_brutes = valeurs_str.split()
            
            try:
                # Conversion des chaînes de caractères en nombres décimaux
                valeurs = [float(x) for x in valeurs_brutes]
            except ValueError:
                continue # Ignore la ligne si la conversion échoue (ex: texte)
                
            if valeurs:
                # Conversion de la liste en tenseur PyTorch 1D
                corpus[utt_id] = torch.tensor(valeurs, dtype=torch.float32)
                
    return corpus

def filtrer_embeddings_memoire(corpus_en_memoire, id_discussion, nom_locuteur):
    """ 
    Filtre directement en RAM depuis le dictionnaire PKL ou CSV 
    """
    embeddings_filtres = {}
    for utt_id, tensor in corpus_en_memoire.items():
        if id_discussion in utt_id and nom_locuteur in utt_id:
            embeddings_filtres[utt_id] = tensor.squeeze()
            
     # On renvoie le gros dictionnaire contenant tous les vecteurs prêts à être filtrés       
    return dict(sorted(embeddings_filtres.items()))

# ==========================================
# 2. Fonctions de Calcul Mathématique
# ==========================================
def calculer_intra_discussion(embeddings):
    """
    Calcule :
    - la matrice de similarité intra-discussion
    - les scores uniques (hors diagonale)
    - la moyenne des similarités
    """
    noms = list(embeddings.keys())
    n = len(noms)
    matrice_sim = np.zeros((n, n))
    
    # Calcul de la matrice complète
    for i in range(n):
        for j in range(n):
            matrice_sim[i, j] = F.cosine_similarity(embeddings[noms[i]], embeddings[noms[j]], dim=0).item()
    
    # Scores uniques (combinaisons 2 à 2)        
    scores_uniques = [F.cosine_similarity(emb1, emb2, dim=0).item() for emb1, emb2 in combinations(embeddings.values(), 2)]
    moyenne_intra = np.mean(scores_uniques) if scores_uniques else 1.0
    
    return matrice_sim, scores_uniques, moyenne_intra, n, noms

def calculer_inter_discussion(emb_ref, emb_cible):
    """
    Calcule la similarité entre deux discussions :
    produit cartésien des embeddings.
    """
    if not emb_ref or not emb_cible:
        return [], 0.0
        
    scores = [F.cosine_similarity(ref, cible, dim=0).item() for ref, cible in product(emb_ref.values(), emb_cible.values())]
    return scores, np.mean(scores)

# ==========================================
# 3. Visualisations et Exportations
# ==========================================
# MODIFICATION : Ajout de nom_locuteur
def generer_rapports(matrice_intra, scores_intra, moyenne_intra, resultats_inter, nom_ref, noms_ref, nom_locuteur, dossier_sortie="resultats_analyse"):
    """
    Génère :
    - heatmap intra-discussion
    - barres horizontales inter-discussion
    - CSV récapitulatif
    """

    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # On conserve le timestamp pour les données du CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    sns.set_theme(style="whitegrid")
    
    # --- A. La Matrice Analytique intra-discussion  ---
    plt.figure(figsize=(10, 8))
    n = matrice_intra.shape[0]
    
    masque_symetrie = np.tril(np.ones_like(matrice_intra, dtype=bool), k=-1)
    limites = [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmap_paliers = mcolors.ListedColormap(sns.color_palette("viridis", n_colors=len(limites) - 1))
    norm_paliers = mcolors.BoundaryNorm(limites, cmap_paliers.N)
    
    ax = sns.heatmap(matrice_intra, annot=False, mask=masque_symetrie, cmap=cmap_paliers, norm=norm_paliers, 
                     cbar_kws={"ticks": limites}, xticklabels=noms_ref, yticklabels=noms_ref)
    ax.invert_yaxis() 
    plt.title(f"Matrice Intra - Réf: {nom_ref}\n(Moyenne hors diagonale: {moyenne_intra:.3f})")
    
    plt.yticks(rotation=0, fontsize=8) 
    plt.xticks(rotation=45, ha='right', fontsize=8) 
    
    plt.tight_layout()
    # MODIFICATION : Utilisation de nom_ref
    plt.savefig(f"{dossier_sortie}/01_matrice_intra_{nom_ref}.png", dpi=300)
    plt.close()

    # --- B. Le Bâton Gradué Horizontal inter-discussion  ---
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
    # MODIFICATION : Utilisation de nom_ref
    plt.savefig(f"{dossier_sortie}/02_moyennes_horizontales_{nom_ref}.png", dpi=300)
    plt.close()

    # --- C. Export CSV Avancé ---
    # MODIFICATION : Utilisation de nom_ref
    fichier_csv = f"{dossier_sortie}/rapport_similitudes_{nom_ref}.csv"
    with open(fichier_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        
        # En-têtes enrichies
        writer.writerow([
            "Type Analyse", "Date Exécution", "Locuteur Cible", "ID Discussion", 
            "Nb Segments", "Paires Comparées", "Similitude Moyenne", 
            "Score Min", "Score Max", "Ecart-Type"
        ])
        
        # Calculs statistiques pour l'INTRA (Référence)
        nb_paires_intra = len(scores_intra)
        ecart_type_intra = np.std(scores_intra) if scores_intra else 0.0
        min_intra = min(scores_intra) if scores_intra else 1.0
        max_intra = max(scores_intra) if scores_intra else 1.0
        
        writer.writerow([
            "INTRA (Réf)", timestamp, nom_locuteur, nom_ref, 
            n, nb_paires_intra, round(moyenne_intra, 4),
            round(min_intra, 4), round(max_intra, 4), round(ecart_type_intra, 4)
        ])
        
        # Calculs statistiques et écriture pour les cibles (INTER)
        for id_cible, data in reversed(cibles_triees):
            scores_inter = data["scores"]
            nb_paires_inter = len(scores_inter)
            
            ecart_type_inter = np.std(scores_inter) if scores_inter else 0.0
            min_inter = min(scores_inter) if scores_inter else data["moyenne"]
            max_inter = max(scores_inter) if scores_inter else data["moyenne"]
            
            writer.writerow([
                "INTER", timestamp, nom_locuteur, id_cible, 
                data["nb_segments"], nb_paires_inter, round(data["moyenne"], 4),
                round(min_inter, 4), round(max_inter, 4), round(ecart_type_inter, 4)
            ])


# ==========================================
# Exécution Principale
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURATION MANUELLE ---
    # MODIFICATION ICI : Tu peux maintenant mettre le chemin vers ton .csv ou .tsv
    SOURCE_DONNEES = "data/embeddings_nettoyes.csv" 
    FICHIER_JSON_PYANNOTE = "segmentation_audios_nettoyes.json"
    LOCUTEUR_CIBLE = "operateur"

    print("--- ETAPE 1 : Analyse des discussions ---")
    discussions_valides = analyser_discussions_valides(FICHIER_JSON_PYANNOTE, LOCUTEUR_CIBLE)
    
    if not discussions_valides:
        print("Aucune discussion valide trouvée pour ce locuteur. Arrêt.")
        exit()

    print(f"\n✅ {len(discussions_valides)} discussions contenant l'opérateur ont été trouvées :")

    # Affichage des discussions trouvées
    liste_ids = list(discussions_valides.keys())
    
    for i, id_disc in enumerate(liste_ids):
        print(f"  [{i+1}] {id_disc} ({discussions_valides[id_disc]} segments)")

    # Choix de la discussion de référence
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
    # --- Chargement des embeddings ---
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

    # Wrapper pour appeler la bonne fonction de chargement dynamiquement
    def extraire(id_disc):
        if est_format_pkl or est_format_csv:
            return filtrer_embeddings_memoire(corpus_en_memoire, id_disc, LOCUTEUR_CIBLE)
        else:
            return charger_segments_tries_pt(SOURCE_DONNEES, id_disc, LOCUTEUR_CIBLE)

    # ----------------------------------------------------
    # ETAPE 3 : Calculs et Graphiques
    # ----------------------------------------------------

    # --- Calculs ---
    print(f"\n📊 Analyse de la référence (Intra-discussion)...")
    emb_ref = extraire(DISCUSSION_REFERENCE)
    
    if len(emb_ref) < 2:
        print("❌ Erreur : Pas assez de segments trouvés dans les embeddings.")
    else:
        matrice_intra, scores_intra, moyenne_intra, nb_seg_ref, noms_ref = calculer_intra_discussion(emb_ref)
        print(f" -> {nb_seg_ref} segments traités. Moyenne intra : {moyenne_intra:.4f}")

        resultats_inter = {}
        print(f"\n📊 Analyse avec la référence (Inter-discussion)...")
        for id_cible in DISCUSSIONS_A_COMPARER:
            emb_cible = extraire(id_cible)
            if emb_cible:
                scores_inter, moyenne_inter = calculer_inter_discussion(emb_ref, emb_cible)
                # MODIFICATION ICI : On ajoute "scores" au dictionnaire
                resultats_inter[id_cible] = {
                    "moyenne": moyenne_inter, 
                    "nb_segments": len(emb_cible),
                    "scores": scores_inter 
                }
                print(f" -> Comparaison {id_cible} ({len(emb_cible)} segs) : {moyenne_inter:.4f}")
        
        # --- Génération des rapports ---
        dossier_dynamique = f"resultats_analyse/res-{DISCUSSION_REFERENCE}"
        
        generer_rapports(
            matrice_intra, 
            scores_intra, 
            moyenne_intra, 
            resultats_inter, 
            DISCUSSION_REFERENCE, 
            noms_ref, 
            LOCUTEUR_CIBLE,
            dossier_sortie=dossier_dynamique
        )
        print(f"\n🎉 Terminé ! Consultez le dossier '{dossier_dynamique}'.")
