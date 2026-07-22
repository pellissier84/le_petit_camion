#!/usr/bin/env python3
"""
Extraction automatique des intervenants depuis un fichier ELAN (.eaf)
- Menu interactif pour le choix des tiers
- SAUVEGARDE des petits fichiers .wav (segments) avec le numéro de l'appel
- CONCATÉNATION en un gros fichier .wav par locuteur
"""

# prerequis : python3 et ffmpeg
# Ce script lit un fichier .eaf, affiche les tiers disponibles, et permet à l'utilisateur de choisir lesquels extraire.
# Il offre aussi l'option de séparer les annotations par valeur (ex: pour un tier "Ac_ev", créer des fichiers distincts pour "und", "bruit", "musique", etc.).
# Les segments audio correspondants sont extraits du fichier source et concaténés en un seul fichier par tier/valeur.
# Les fichiers résultants sont nommés de manière claire (ex: "Ac_ev_und.wav", "Ac_ev_bruit.wav", etc.).
# conda activate audio_env
# commande de lancement : python extraire_elan.py mon_corpus.eaf enregistrement.wav
# exemple python extraire_elan_diff_tier.py audio_elan_3540.eaf audio_elan_3540.wav
# exemple python extraire_elan_diff_tier.py audio-1775033540.32214.eaf audio-1775033540.32214.wav
# audio-1775031826.41778.wav
# audio-1775031968.41843.wav
# audio-1775033540.32214.wav

import xml.etree.ElementTree as ET
import subprocess
import os
import sys
import re
from collections import defaultdict

def extraire_tous_les_tiers(fichier_eaf):
    """Extrait la liste de tous les tiers disponibles"""
    arbre = ET.parse(fichier_eaf)
    racine = arbre.getroot()
    tiers = set()
    for tier in racine.findall('.//TIER'):
        nom_tier = tier.get('TIER_ID')
        if nom_tier:
            tiers.add(nom_tier)
    return sorted(list(tiers))

def demander_tiers_a_extraire(tous_les_tiers):
    """Demande à l'utilisateur quels tiers extraire"""
    print("\n" + "="*50)
    print("TIERS DISPONIBLES :")
    print("="*50)
    for i, tier in enumerate(tous_les_tiers, 1):
        print(f"  {i}. {tier}")
    
    print("\n" + "-"*50)
    print("Options :")
    print("  - Entrez les numéros séparés par des virgules (ex: 1,3)")
    print("  - Entrez une plage (ex: 1-4)")
    print("  - Tapez 'all' pour extraire TOUS les tiers")
    print("  - Tapez 'q' pour quitter")
    print("-" * 50)
    
    choix = input("\nVotre choix : ").strip().lower()
    
    if choix == 'q':
        return None
    if choix == 'all':
        return tous_les_tiers
        
    selection = set()
    if '-' in choix and ',' not in choix:
        debut, fin = choix.split('-')
        for i in range(int(debut), int(fin)+1):
            if 1 <= i <= len(tous_les_tiers):
                selection.add(tous_les_tiers[i-1])
    else:
        for partie in choix.split(','):
            partie = partie.strip()
            if partie.isdigit():
                i = int(partie)
                if 1 <= i <= len(tous_les_tiers):
                    selection.add(tous_les_tiers[i-1])
                    
    if not selection:
        return tous_les_tiers
    return list(selection)

def demander_separation_par_valeur(tiers_selectionnes):
    """Demande si on veut séparer par valeur d'annotation pour certains tiers"""
    print("\n" + "-"*50)
    print("SÉPARATION PAR SOUS-CATÉGORIES ?")
    print("-" * 50)
    for i, tier in enumerate(tiers_selectionnes, 1):
        print(f"  {i}. {tier}")
        
    print("\nEntrez les numéros des tiers à séparer par valeur (ex: 1)")
    print("Ou appuyez sur Entrée pour ignorer")
    
    choix = input("\nVotre choix : ").strip()
    if not choix:
        return []
        
    selection = []
    for partie in choix.split(','):
        partie = partie.strip()
        if partie.isdigit():
            i = int(partie) - 1
            if 0 <= i < len(tiers_selectionnes):
                selection.append(tiers_selectionnes[i])
    return selection

def extraire_temps_depuis_eaf(fichier_eaf, tiers_selectionnes, separator_par_valeur=False):
    """Extrait les temps, avec option de séparation par valeur d'annotation"""
    arbre = ET.parse(fichier_eaf)
    racine = arbre.getroot()
    
    time_slots = {}
    for ts in racine.findall('.//TIME_ORDER'):
        for tsi in ts.findall('TIME_SLOT'):
            tid = tsi.get('TIME_SLOT_ID')
            tval = tsi.get('TIME_VALUE')
            time_slots[tid] = int(tval) / 1000.0
            
    if separator_par_valeur:
        annotations = defaultdict(lambda: defaultdict(list))
    else:
        annotations = defaultdict(list)
        
    for tier in racine.findall('.//TIER'):
        nom_tier = tier.get('TIER_ID')
        if nom_tier not in tiers_selectionnes:
            continue
            
        for annotation in tier.findall('.//ALIGNABLE_ANNOTATION'):
            ts_debut = annotation.get('TIME_SLOT_REF1')
            ts_fin = annotation.get('TIME_SLOT_REF2')
            valeur = annotation.find('ANNOTATION_VALUE').text or ""
            
            valeur_propre = valeur.strip().lower()
            if not valeur_propre:
                valeur_propre = "sans_valeur"
                
            if ts_debut in time_slots and ts_fin in time_slots:
                segment = {
                    'debut': time_slots[ts_debut],
                    'fin': time_slots[ts_fin],
                    'valeur': valeur
                }
                if separator_par_valeur:
                    annotations[nom_tier][valeur_propre].append(segment)
                else:
                    annotations[nom_tier].append(segment)
                    
    # Tri par ordre chronologique
    if separator_par_valeur:
        for tier in annotations:
            for valeur in annotations[tier]:
                annotations[tier][valeur].sort(key=lambda x: x['debut'])
    else:
        for tier in annotations:
            annotations[tier].sort(key=lambda x: x['debut'])
            
    return annotations

def extraire_segment_ffmpeg(audio_source, debut, fin, fichier_sortie):
    """Extrait un segment audio avec ffmpeg (en 16000Hz pour l'IA)"""
    duree = fin - debut
    cmd = [
        'ffmpeg', '-i', audio_source,
        '-ss', str(debut), '-t', str(duree),
        '-acodec', 'pcm_s16le', '-ar', '16000',
        '-y', fichier_sortie
    ]
    # On masque les logs pour ne pas saturer le terminal
    resultat = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return resultat.returncode == 0

def concatener_segments(segments, fichier_sortie):
    """Concatène des segments audio avec ffmpeg"""
    if not segments:
        return False
        
    with open('temp_concat_list.txt', 'w') as f:
        for seg in segments:
            f.write(f"file '{seg}'\n")
            
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', 'temp_concat_list.txt', '-c', 'copy',
        '-y', fichier_sortie
    ]
    resultat = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists('temp_concat_list.txt'):
        os.remove('temp_concat_list.txt')
    return resultat.returncode == 0

def main():
    if len(sys.argv) != 3:
        print("Usage: python extraire_elan.py <fichier.eaf> <audio.wav>")
        sys.exit(1)
        
    fichier_eaf = sys.argv[1]
    audio_source = sys.argv[2]
    
    if not os.path.exists(fichier_eaf) or not os.path.exists(audio_source):
        print(f"❌ Erreur: {fichier_eaf} ou {audio_source} n'existe pas")
        sys.exit(1)
        
    # Extraction de l'ID de l'appel depuis le nom du fichier wav (ex: "audio_elan_3540.wav" -> "3540")
    nom_base_audio = os.path.basename(audio_source)
    match_id = re.search(r'\d+', nom_base_audio)
    appel_id = match_id.group(0) if match_id else "seq"
        
    print(f"\n📁 Lecture de {fichier_eaf}...")
    print(f"🆔 ID d'appel détecté : {appel_id}")
    
    tous_les_tiers = extraire_tous_les_tiers(fichier_eaf)
    if not tous_les_tiers:
        print("❌ Aucun tier trouvé")
        sys.exit(1)
        
    tiers_a_extraire = demander_tiers_a_extraire(tous_les_tiers)
    if tiers_a_extraire is None:
        sys.exit(0)
        
    tiers_a_separer = demander_separation_par_valeur(tiers_a_extraire)
    
    # Création du dossier PERMANENT pour les segments individuels
    dossier_segments = "segments_extraits"
    os.makedirs(dossier_segments, exist_ok=True)
    
    annotations_separees = {}
    for tier in tiers_a_extraire:
        if tier in tiers_a_separer:
            ann = extraire_temps_depuis_eaf(fichier_eaf, [tier], separator_par_valeur=True)
            if tier in ann:
                annotations_separees[tier] = ann[tier]
        else:
            ann = extraire_temps_depuis_eaf(fichier_eaf, [tier], separator_par_valeur=False)
            if tier in ann:
                annotations_separees[tier] = {"_total_": ann[tier]}
                
    resultats_complets = []
    total_petits_fichiers = 0
    
    for tier, sous_categories in annotations_separees.items():
        for valeur, segments in sous_categories.items():
            if not segments:
                continue
                
            if valeur == "_total_":
                nom_fichier_concat = f"{tier}_complet_{appel_id}.wav"
                nom_affichage = tier
                valeur_propre = appel_id # On utilise l'ID (ex: 3540) au lieu de "seq"
            else:
                nom_propre = valeur.replace(' ', '_').replace('/', '_')
                nom_fichier_concat = f"{tier}_{appel_id}_{nom_propre}.wav"
                nom_affichage = f"{tier} [{valeur}]"
                valeur_propre = f"{appel_id}_{nom_propre}"
                
            print(f"\n🎤 Traitement de {nom_affichage} ({len(segments)} segments)...")
            
            fichiers_segments = []
            # On commence l'énumération à 1 pour avoir 001, 002, etc.
            for i, seg in enumerate(segments, 1):
                debut = seg['debut']
                fin = seg['fin']
                
                # On sauvegarde dans le dossier segments_extraits
                fichier_segment = f"{dossier_segments}/{tier}_{valeur_propre}_{i:03d}.wav"
                
                if extraire_segment_ffmpeg(audio_source, debut, fin, fichier_segment):
                    fichiers_segments.append(fichier_segment)
                    total_petits_fichiers += 1
                    
            # Si on a réussi à extraire des segments, on les concatène
            if fichiers_segments:
                print(f"  -> Concaténation en cours...")
                if concatener_segments(fichiers_segments, nom_fichier_concat):
                    print(f"  ✅ {nom_fichier_concat} créé !")
                    resultats_complets.append(nom_fichier_concat)
                    
    print("\n" + "="*50)
    print("✅ TRAITEMENT TOTALEMENT TERMINÉ !")
    print("="*50)
    print(f"-> {total_petits_fichiers} petits fichiers sauvegardés dans '{dossier_segments}/'")
    print("-> Fichiers concaténés générés à la racine :")
    for r in sorted(resultats_complets):
        print(f"   📁 {r}")
    print("="*50)

if __name__ == "__main__":
    main()
