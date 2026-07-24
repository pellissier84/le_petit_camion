import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

def analyser_croisement_clusters(dossier_clusters, dossier_sortie="resultats_croisement"):
    """
    Analyse avancée des clusters audio.

    Objectifs :
    - Lire tous les fichiers cluster_locuteur_*.json
    - Construire une matrice de co-occurrence (référence → membres)
    - Détecter :
        1. Réciprocités simples (A ↔ B)
        2. Asymétries / dominances (A → B sans retour)
        3. Triades renforcées (A ↔ B et A → C → B)
    - Générer une heatmap multi-niveaux
    - Exporter un CSV + une image + une synthèse console
    """
    chemin_entree = Path(dossier_clusters)
    
    # Recherche dans tous les sous-dossiers
    fichiers_json = list(chemin_entree.rglob('cluster_locuteur_*.json'))
    
    if not fichiers_json:
        print(f"❌ Aucun fichier JSON trouvé dans {dossier_clusters} ni dans ses sous-dossiers.")
        return
        
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 1. Collecte des données
    donnees_references = {}
    tous_les_audios = set()  # liste globale des discussions
    
    for fichier in fichiers_json:
        with open(fichier, 'r', encoding='utf-8') as f:
            data = json.load(f)
            ref = data.get("reference_utilisee")
            membres = data.get("membres_cluster", [])
            
            if ref:
                donnees_references[ref] = set(membres)
                tous_les_audios.add(ref)
                tous_les_audios.update(membres)
                
    # Liste triée des discussions            
    liste_audios = sorted(list(tous_les_audios))
    
    # 2. Création de la matrice de co-occurrence de base (0 et 1)
    df_cooccurrence = pd.DataFrame(0, index=liste_audios, columns=liste_audios)
    
    # Remplissage : 1 si ref inclut membre
    for ref, membres in donnees_references.items():
        for membre in membres:
            df_cooccurrence.at[ref, membre] = 1
            
    # Filtrage des lignes/colonnes vides
    df_filtre = df_cooccurrence.loc[df_cooccurrence.sum(axis=1) > 0]
    df_filtre = df_filtre.loc[:, df_filtre.sum(axis=0) > 0]
    
    if df_filtre.empty:
        print("⚠️ Après filtrage, il n'y a aucune connexion entre les discussions. Aucune matrice générée.")
        return

    # ==========================================
    # 3. DÉTECTION DES RELATIONS COMPLEXES (Asymétries et Triades)
    # ==========================================
    df_affichage = df_filtre.copy()
    
    for ref in df_affichage.index:      # ref = Audio A
        for cible in df_affichage.columns: # cible = Audio B
            
            if df_filtre.at[ref, cible] == 1: # Si A inclut B
                
                # Vérification 1 : La réciprocité (B inclut-il A ?)
                capture_reciproque = False
                if cible in df_filtre.index and ref in df_filtre.columns:
                    if df_filtre.at[cible, ref] == 1:
                        capture_reciproque = True
                
                # Si non réciproque -> Asymétrique / Dominant (Valeur 2)
                if not capture_reciproque:
                    df_affichage.at[ref, cible] = 2
                    
                # Si réciproque, Vérification 2 : La Triade (Existe-t-il un C ?)
                else:
                    triade_trouvee = False
                    
                    for c in df_filtre.columns:
                        # C doit être un audio différent de A et de B
                        if c != ref and c != cible:
                            # A inclut C ?
                            a_inclut_c = (c in df_filtre.columns and ref in df_filtre.index and df_filtre.at[ref, c] == 1)
                            # C inclut B ?
                            c_inclut_b = (cible in df_filtre.columns and c in df_filtre.index and df_filtre.at[c, cible] == 1)
                            
                            if a_inclut_c and c_inclut_b:
                                triade_trouvee = True
                                break # On a trouvé un C valide, on arrête de chercher
                                
                    if triade_trouvee:
                        df_affichage.at[ref, cible] = 3 # Triade / Réciprocité renforcée (Valeur 3)
                    else:
                        df_affichage.at[ref, cible] = 1 # Réciprocité simple (Valeur 1)

    # 4. Export de la matrice brute (données 0/1) en CSV
    chemin_csv = f"{dossier_sortie}/matrice_cooccurrence_filtree_{timestamp}.csv"
    df_filtre.to_csv(chemin_csv, sep=';')
    print(f"✅ Matrice CSV exportée : {chemin_csv}")
    
    # 5. Génération de la Heatmap multi-niveaux
    plt.figure(figsize=(max(14, len(df_affichage.columns)*0.5), max(10, len(df_affichage.index)*0.5)))
    
    # Palette à 4 niveaux :
    # 0 = Blanc (Rien)
    # 1 = Bleu (Réciproque simple)
    # 2 = Orange (Asymétrique / Dominant)
    # 3 = Violet (Triade / Renforcée par un tiers)
    couleurs = ["#ffffff", "#3498db", "#e67e22", "#9b59b6"] 
    cmap_perso = mcolors.ListedColormap(couleurs)
    
    ax = sns.heatmap(df_affichage, 
                     cmap=cmap_perso, 
                     linewidths=0.5, 
                     linecolor='lightgray', 
                     cbar=False, 
                     square=True,
                     vmin=0, vmax=3) # Force l'échelle de 0 à 3
                     
    # Griser une colonne sur deux (Effet Zébré)
    for i in range(1, len(df_affichage.columns), 2):
        ax.axvspan(i, i+1, color='gray', alpha=0.15, zorder=2)
        
    plt.title("Matrice de Co-occurrence Multi-Niveaux des Clusters", fontsize=16, fontweight='bold', pad=30)
    plt.xlabel("Audios Capturés (Cibles)", fontsize=12, fontweight='bold')
    plt.ylabel("Audios de Référence (Générateurs)", fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Ajout de la légende détaillée
    patch_rec = mpatches.Patch(color='#3498db', label='Réciproque Simple (A ↔ B)')
    patch_asym = mpatches.Patch(color='#e67e22', label='Dominant (A → B, sans retour)')
    patch_triade = mpatches.Patch(color='#9b59b6', label='Triade Renforcée (A ↔ B, et ∃ C t.q. A → C → B)')
    
    plt.legend(handles=[patch_rec, patch_asym, patch_triade], 
               loc='upper right', bbox_to_anchor=(1.0, 1.12), fontsize=10, frameon=True)
    
    # Bordure globale
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    chemin_img = f"{dossier_sortie}/heatmap_croisement_triade_{timestamp}.png"
    plt.savefig(chemin_img, dpi=300)
    plt.close()
    print(f"✅ Carte de chaleur (zébrée avec asymétries et triades) exportée : {chemin_img}")
    
    # 6. Synthèse console
    print("\n--- SYNTHÈSE DES REGROUPEMENTS ---")
    for ref in df_affichage.index:
        nb_recip = (df_affichage.loc[ref] == 1).sum()
        nb_asym = (df_affichage.loc[ref] == 2).sum()
        nb_triade = (df_affichage.loc[ref] == 3).sum()
        print(f"🎙️ Réf: {ref}")
        print(f"   ↳ Réciproques : {nb_recip} | Dominants : {nb_asym} | Triades : {nb_triade}")

if __name__ == "__main__":
    DOSSIER_CLUSTERS = "resultats_analyse" 
    analyser_croisement_clusters(DOSSIER_CLUSTERS)
