import json
import pympi
import matplotlib.pyplot as plt

# comparaison visuelle des references eaf par rapport au wav decoupes par pyannote
# affichage des locuteurs
# en utilisant : segmentation_audios_Nexsis.json
# les eaf
# audio-1775031826.41778.eaf
# audio-1775031968.41843.eaf
# audio-1775033540.32214.eaf
# les wav ont meme nom


def extraire_donnees_elan(chemin_eaf):
    """Extrait les segments de parole d'un fichier ELAN (.eaf)"""
    eaf = pympi.Elan.Eaf(chemin_eaf)
    segments_elan = []
    
    # Dans ELAN, on suppose généralement qu'il y a une "tier" (piste) par locuteur
    for nom_tier in eaf.get_tier_names():
        annotations = eaf.get_annotation_data_for_tier(nom_tier)
        for ann in annotations:
            # ELAN stocke le temps en millisecondes, on convertit en secondes
            debut = ann[0] / 1000.0
            fin = ann[1] / 1000.0
            segments_elan.append({'start': debut, 'end': fin, 'speaker': nom_tier})
            
    return segments_elan

def extraire_donnees_json(chemin_json, nom_fichier_wav):
    """Extrait les segments d'un fichier audio spécifique depuis le JSON Pyannote"""
    with open(chemin_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for fichier in data['files']:
        if fichier['file'] == nom_fichier_wav:
            return fichier.get('turns', [])
            
    print(f"Attention: {nom_fichier_wav} non trouvé dans le JSON.")
    return []

def tracer_comparaison(segments_ref, segments_hyp, titre):
    """Trace deux chronogrammes l'un au-dessus de l'autre"""
    
    # Extraire les locuteurs uniques pour attribuer les couleurs
    locuteurs_ref = sorted(list(set([s['speaker'] for s in segments_ref])))
    locuteurs_hyp = sorted(list(set([s['speaker'] for s in segments_hyp])))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    fig.suptitle(f'Comparaison de la Diarisation : {titre}', fontsize=14, fontweight='bold')

    # --- Graphique du Haut : Vérité Terrain (ELAN) ---
    couleurs_ref = plt.cm.get_cmap('Set1', max(3, len(locuteurs_ref)))
    dico_c_ref = {loc: couleurs_ref(i) for i, loc in enumerate(locuteurs_ref)}
    
    for seg in segments_ref:
        y_pos = locuteurs_ref.index(seg['speaker'])
        ax1.broken_barh([(seg['start'], seg['end'] - seg['start'])], (y_pos - 0.4, 0.8), 
                        facecolors=dico_c_ref[seg['speaker']], edgecolor='black', linewidth=0.5)
    
    ax1.set_title('Vérité Terrain (Annotations Manuelles ELAN)')
    ax1.set_yticks(range(len(locuteurs_ref)))
    ax1.set_yticklabels(locuteurs_ref)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)

    # --- Graphique du Bas : Prédictions (Pyannote JSON) ---
    couleurs_hyp = plt.cm.get_cmap('Set2', max(3, len(locuteurs_hyp)))
    dico_c_hyp = {loc: couleurs_hyp(i) for i, loc in enumerate(locuteurs_hyp)}
    
    for seg in segments_hyp:
        y_pos = locuteurs_hyp.index(seg['speaker'])
        ax2.broken_barh([(seg['start'], seg['end'] - seg['start'])], (y_pos - 0.4, 0.8), 
                        facecolors=dico_c_hyp[seg['speaker']], edgecolor='black', linewidth=0.5)
    
    ax2.set_title('Prédictions (Modèle Pyannote)')
    ax2.set_yticks(range(len(locuteurs_hyp)))
    ax2.set_yticklabels(locuteurs_hyp)
    ax2.set_xlabel('Temps (secondes)')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# ==========================================
# EXÉCUTION DU SCRIPT (À modifier avec vos chemins)
# ==========================================

# 1. Renseignez les chemins vers vos fichiers sur votre ordinateur
CHEMIN_EAF = "audio-1775033540.32214.eaf"
CHEMIN_JSON = "segmentation_audios_Nexsis.json"
NOM_DU_WAV_A_CHERCHER = "audio-1775033540.32214.wav" # Le nom exact tel qu'il est dans le JSON

# 2. Extraction
segments_elan = extraire_donnees_elan(CHEMIN_EAF)
segments_pyannote = extraire_donnees_json(CHEMIN_JSON, NOM_DU_WAV_A_CHERCHER)

# 3. Affichage
if segments_elan and segments_pyannote:
    tracer_comparaison(segments_elan, segments_pyannote, NOM_DU_WAV_A_CHERCHER)
else:
    print("Impossible de tracer la comparaison, des données sont manquantes.")
