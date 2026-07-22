import json
import pympi
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

# rapport metrique entre eaf et wav decoupé suivant les locuteurs de pyannote
# les eaf (meme nom pour les .wav)
# audio-1775031826.41778.eaf
# pour cet audio wac coupé a 4mn, et enlevement du tier Ac_ev
# audio-1775031968.41843.eaf
# audio-1775033540.32214.eaf
# fichier des decoupes par pyannote de 116 wav
# segmentation_audios_Nexsis.json

def charger_verite_elan(chemin_eaf, tiers_a_ignorer):
    """Extrait les segments ELAN en ignorant les pistes de bruit"""
    eaf = pympi.Elan.Eaf(chemin_eaf)
    annotation = Annotation(uri="mon_audio")
    
    for nom_tier in eaf.get_tier_names():
        # On saute cette piste si elle fait partie de notre liste noire
        if nom_tier in tiers_a_ignorer:
            continue
            
        annotations = eaf.get_annotation_data_for_tier(nom_tier)
        for ann in annotations:
            debut, fin = ann[0] / 1000.0, ann[1] / 1000.0
            annotation[Segment(debut, fin)] = nom_tier
            
    return annotation

def charger_prediction_json(chemin_json, nom_etiquette_audio):
    annotation = Annotation(uri="mon_audio")
    with open(chemin_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for fichier in data['files']:
        if fichier['file'] == nom_etiquette_audio:
            for seg in fichier.get('turns', []):
                annotation[Segment(seg['start'], seg['end'])] = seg['speaker']
            return annotation
    raise ValueError("Fichier non trouvé dans le JSON.")

# ==========================================
# VOS PARAMÈTRES (À MODIFIER)
# ==========================================

CHEMIN_EAF = "audio-1775033540.32214.eaf" 
CHEMIN_JSON = "segmentation_audios_Nexsis.json" 
NOM_ETIQUETTE_DANS_JSON = "audio-1775033540.32214.wav" 

# --- NOUVEAUX PARAMÈTRES DE CORRECTION ---
# 1. Pistes ELAN à ne pas compter comme de la parole humaine
TIERS_A_IGNORER = ["Ac_ev"] 

# 2. Temps maximum d'évaluation en secondes (4 minutes = 240 secondes)
TEMPS_LIMITE_SECONDES = 240.0 

# ==========================================
# CALCUL DU SCORE
# ==========================================

try:
    reference = charger_verite_elan(CHEMIN_EAF, TIERS_A_IGNORER)
    hypothese = charger_prediction_json(CHEMIN_JSON, NOM_ETIQUETTE_DANS_JSON)
    
    # Création de la zone d'évaluation (UEM) : de 0 à 240 secondes
    zone_evaluation = Timeline([Segment(0.0, TEMPS_LIMITE_SECONDES)])
    
    der_metric = DiarizationErrorRate()
    # On passe notre Timeline au calculateur via le paramètre 'uem'
    details = der_metric(reference, hypothese, uem=zone_evaluation, detailed=True)
    
    total_parole = details['total']
    der_total = details['diarization error rate'] * 100
    fausses_alertes = (details['false alarm'] / total_parole) * 100
    omissions = (details['missed detection'] / total_parole) * 100
    confusions = (details['confusion'] / total_parole) * 100
    
    print("\n" + "="*50)
    print(f"📊 RAPPORT D'ÉVALUATION CORRIGÉ (DER)")
    print(f"Évalué uniquement sur les {TEMPS_LIMITE_SECONDES} premières secondes.")
    print("="*50)
    print(f"DER Global          : {der_total:.2f} % d'erreur au total")
    print("-" * 50)
    print(f"  - Omissions       : {omissions:.2f} %")
    print(f"  - Fausses Alertes : {fausses_alertes:.2f} %")
    print(f"  - Confusions      : {confusions:.2f} %")
    print("="*50 + "\n")

except Exception as e:
    print(f"Une erreur est survenue : {e}")
