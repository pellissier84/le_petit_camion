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
    Analyse tous les clusters générés par l’analyse audio.
    
    Objectifs :
    - Lire tous les fichiers cluster_locuteur_*.json
    - Construire une matrice de co-occurrence (référence → membres)
    - Détecter les relations asymétriques (A inclut B mais B n’inclut pas A)
    - Générer une heatmap zébrée pour visualiser les relations
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
    tous_les_audios = set()   # liste globale des discussions
    
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
    
    # 2. Création de la matrice de co-occurrence de base
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
    #  Détection des asymétries
    # ==========================================
    # On crée une copie pour l'affichage (afin de ne pas corrompre le CSV avec la valeur 2)
    df_affichage = df_filtre.copy()
    
    for ref in df_affichage.index:
        for cible in df_affichage.columns:
            if df_affichage.at[ref, cible] == 1:
                capture_reciproque = False
                
                # On vérifie si la cible a aussi été une référence qui a capturé 'ref'
                # Il faut s'assurer que 'cible' existe en index et 'ref' en colonne
                if cible in df_affichage.index and ref in df_affichage.columns:
                    if df_filtre.at[cible, ref] == 1:
                        capture_reciproque = True
                
                # Si A capture B, mais B ne capture pas A -> Valeur 2 (Asymétrique)
                if not capture_reciproque:
                    df_affichage.at[ref, cible] = 2

    # 3. Export de la matrice (données brutes 0/1) en CSV
    chemin_csv = f"{dossier_sortie}/matrice_cooccurrence_filtree_{timestamp}.csv"
    df_filtre.to_csv(chemin_csv, sep=';')
    print(f"✅ Matrice CSV exportée : {chemin_csv}")
    
    # 4. Génération de la Heatmap
    plt.figure(figsize=(max(12, len(df_affichage.columns)*0.5), max(9, len(df_affichage.index)*0.5)))
    
    # Palette personnalisée : 0=Blanc, 1=Bleu (Réciproque), 2=Orange (Asymétrique)
    couleurs = ["#ffffff", "#3498db", "#e67e22"] 
    cmap_perso = mcolors.ListedColormap(couleurs)
    
    ax = sns.heatmap(df_affichage, 
                     cmap=cmap_perso, 
                     linewidths=0.5, 
                     linecolor='lightgray', 
                     cbar=False, # On désactive la barre par défaut pour créer une légende propre
                     square=True)
                     
    # Griser une colonne sur deux (Effet Zébré)
    for i in range(1, len(df_affichage.columns), 2):
        ax.axvspan(i, i+1, color='gray', alpha=0.15, zorder=2)
        
    plt.title("Matrice de Co-occurrence des Clusters", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Audios Capturés (Cibles)", fontsize=12, fontweight='bold')
    plt.ylabel("Audios de Référence (Générateurs)", fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Ajout de la légende personnalisée
    patch_sym = mpatches.Patch(color='#3498db', label='Réciproque (A inclut B, et B inclut A)')
    patch_asym = mpatches.Patch(color='#e67e22', label='Dominant (A inclut B, mais B n\'inclut pas A)')
    plt.legend(handles=[patch_sym, patch_asym], loc='upper right', bbox_to_anchor=(1.0, 1.08), fontsize=10, frameon=True)
    
    # Bordure globale
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    chemin_img = f"{dossier_sortie}/heatmap_croisement_avancee_{timestamp}.png"
    plt.savefig(chemin_img, dpi=300)
    plt.close()
    print(f"✅ Carte de chaleur (zébrée avec asymétries) exportée : {chemin_img}")
    
    # 5. Synthèse console
    print("\n--- SYNTHÈSE DES REGROUPEMENTS ---")
    for ref in df_filtre.index:
        membres = df_filtre.columns[df_filtre.loc[ref] == 1].tolist()
        print(f"🎙️ Réf: {ref}")
        print(f"   ↳ {len(membres)} connexions : {', '.join(membres)}")

if __name__ == "__main__":
    DOSSIER_CLUSTERS = "resultats_analyse" 
    analyser_croisement_clusters(DOSSIER_CLUSTERS)
