import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import re
import warnings
import scipy.signal as scipy_signal
warnings.filterwarnings('ignore')

# ============================================
# SCRIPT DE CLASSIFICATION DE LOCUTEURS
# Optimisé pour les corpus d'appels (ex: SDIS/Nexsis)
# ============================================

# ============================================
# 1. FONCTIONS D'EXTRACTION AMÉLIORÉES
# ============================================

def pretraiter_audio(signal_np, fs):
  """
    Prétraitement du signal audio pour améliorer la qualité avant extraction.
    - Normalisation RMS pour homogénéiser les volumes.
    - Filtre passe-haut 80 Hz pour supprimer les bourdonnements graves.
    - Préaccentuation pour renforcer les hautes fréquences utiles à la voix.
    """
    # 1. Normalisation du volume (RMS) pour lisser les différences de gain
    rms = np.sqrt(np.mean(signal_np**2))
    if rms > 1e-6:
        signal_np = signal_np / rms * 0.1
    
    # 2. Filtre passe-haut à 80Hz : élimine les bourdonnements graves (bruit électrique/vent)
    try:
        b, a = scipy_signal.butter(4, 80 / (fs/2), 'high')
        signal_np = scipy_signal.filtfilt(b, a, signal_np)
    except:
		# En cas d'erreur (fichier trop court), on ignore le filtrage
        pass
    
    # 3. Préaccentuation : amplifie les hautes fréquences vocales (rend la voix plus claire pour le modèle)
    preemph = 0.97
    signal_np = np.append(signal_np[0], signal_np[1:] - preemph * signal_np[:-1])
    
    return signal_np


def detecter_parole(signal_np, fs, seuil_energie=0.005):
    """
    Détecteur d'activité vocale (VAD) basé sur l'énergie.
    - Découpe le signal en fenêtres de 25 ms.
    - Calcule l'énergie de chaque fenêtre.
    - Détermine si la fenêtre contient de la parole.
    """
    window_size = int(0.025 * fs) # 25 ms
    hop_size = int(0.010 * fs)    # 10 ms
    
    # Si le fichier est trop court, on considère qu'il contient de la parole
    if len(signal_np) < window_size:
        return np.array([True])
    
    # Calcul de l'énergie par fenêtre
    energy = np.array([
        np.sum(signal_np[i:i+window_size]**2) 
        for i in range(0, len(signal_np)-window_size, hop_size)
    ])
    
    if len(energy) == 0:
        return np.array([True])
    
    # Seuil dynamique basé sur l'énergie max
    threshold = seuil_energie * np.max(energy)
    # Frames considérées comme parole
    speech_frames = energy > threshold
    
    return speech_frames

# ============================================================
# 2. EXTRACTION DES EMBEDDINGS (SpeechBrain ECAPA-TDNN)
# ============================================================

def get_embedding_ameliore(audio_path, classifier, max_duration=10, 
                          device='cpu', mode='vad', use_pretraitement=True):
    """
    Extraction de l'empreinte vocale (embedding) via SpeechBrain.
    Propose plusieurs stratégies de sélection des frames pour contourner 
    les fichiers partiellement corrompus ou très bruités avec plusieurs modes:
    - 'vad': détection de parole avec sélection de la zone la plus dense
    - 'top_k': sélection des meilleures frames
    - 'weighted': pondération par énergie
    - 'standard': mode standard
    """
    target_sample_rate = 16000
    signal, fs = torchaudio.load(audio_path)
    
    # Si le fichier est stéréo, on convertit en mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    # Resampling en 16 kHz si nécessaire
    if fs != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sample_rate)
        signal = resampler(signal)
        fs = target_sample_rate
    
    # Conversion en numpy
    signal_np = signal.squeeze().numpy()
    
    # Prétraitement audio (normalisation, filtre, préaccentuation)
    if use_pretraitement:
        signal_np = pretraiter_audio(signal_np, fs)
    
    # --- Modes d'extraction d'embeddings ---
    # ==========================================
    if mode == 'vad':
        # Détection de parole avec sélection de la zone la plus dense
        speech_frames = detecter_parole(signal_np, fs, seuil_energie=0.005)
        
        # On exige au moins 20 frames de parole pour être fiable
        if np.sum(speech_frames) > 20:  # Au moins 20 frames de parole
            window_size = int(0.025 * fs)
            hop_size = int(0.010 * fs)
            speech_indices = np.where(speech_frames)[0]
            
            # Trouver la zone de parole la plus dense (3 secondes)
            if len(speech_indices) > 10:
                window_frames = int(3 * fs / hop_size)
                best_start = 0
                best_count = 0
                
                # On glisse une fenêtre et on compte les frames vocales
                for i in range(0, len(speech_indices) - window_frames, window_frames // 2):
                    count = np.sum(speech_indices[i:i+window_frames])
                    if count > best_count:
                        best_count = count
                        best_start = i
                
                # Détermination des indices temporels
                start_frame = speech_indices[best_start] * hop_size
                end_frame = min(speech_indices[best_start + min(window_frames, len(speech_indices)-best_start-1)] * hop_size + window_size, 
                               len(signal_np))
                
                # On ne garde que la zone dense en parole
                signal_np = signal_np[start_frame:end_frame]
    
    elif mode == 'top_k':
		"""
        Sélection des frames les plus énergétiques.
        On garde les 40% de frames ayant la plus forte énergie.
        """
        # Sélection des K meilleures frames (40% avec plus haute énergie)
        window_size = int(0.025 * fs)
        hop_size = int(0.010 * fs)
        
        if len(signal_np) > window_size:
            energy = np.array([
                np.sum(signal_np[i:i+window_size]**2) 
                for i in range(0, len(signal_np)-window_size, hop_size)
            ])
            
            if len(energy) > 10:
                k = max(10, int(0.4 * len(energy)))
                top_indices = np.argsort(energy)[-k:]
                top_indices.sort()
                
                selected_signal = []
                for idx in top_indices:
                    start = idx * hop_size
                    end = min(start + window_size, len(signal_np))
                    selected_signal.append(signal_np[start:end])
                
                if selected_signal:
                    signal_np = np.concatenate(selected_signal)
    
    elif mode == 'weighted':
		"""
        Pondération des frames par leur énergie.
        Les frames les plus fortes contribuent davantage à l'embedding final.
        """
        # Pondération par énergie
        window_size = int(0.025 * fs)
        hop_size = int(0.010 * fs)
        
        if len(signal_np) > window_size:
            energy = np.array([
                np.sum(signal_np[i:i+window_size]**2) 
                for i in range(0, len(signal_np)-window_size, hop_size)
            ])
            
            if len(energy) > 10:
                weights = energy / (np.sum(energy) + 1e-10)
                
                weighted_signal = []
                for i, w in enumerate(weights):
                    start = i * hop_size
                    end = min(start + window_size, len(signal_np))
                    frame = signal_np[start:end]
                    weighted_signal.append(frame * w)
                
                if weighted_signal:
                    signal_np = np.concatenate(weighted_signal)
    
    # Troncature / padding pour garantir une durée correcte
    # =====================================================
    max_samples = max_duration * fs
    
    # Si trop long → on coupe au centre
    if len(signal_np) > max_samples:
        start = (len(signal_np) - max_samples) // 2
        signal_np = signal_np[start:start + max_samples]
    
    # Si le signal est trop court, le padder
    if len(signal_np) < 0.5 * fs:  # Moins de 0.5s
        signal_np = np.pad(signal_np, (0, int(0.5 * fs) - len(signal_np)))
    
    # Conversion en tensor PyTorch
    signal = torch.from_numpy(signal_np).float().unsqueeze(0)
    signal = signal.to(device)
    
    # Extraction ECAPA-TDNN
    with torch.no_grad():
        embedding = classifier.encode_batch(signal)
    
    # Normalisation L2
    embedding = embedding.squeeze()
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    
    return embedding

# Extraction multi-frames (plus robuste aux variations)
# ======================================================

def get_embedding_multi_frames(audio_path, classifier, device='cpu', 
                               n_frames=3, duration_per_frame=3):
    """
    Extrait plusieurs embeddings sur différentes parties du fichier
    et calcule la moyenne. Rend la signature plus robuste aux bruits sporadiques.
    """
    target_sample_rate = 16000
    signal, fs = torchaudio.load(audio_path)
    
    # Conversion stéréo → mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    # Resampling
    if fs != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sample_rate)
        signal = resampler(signal)
        fs = target_sample_rate
    
    signal_np = signal.squeeze().numpy()
    
    # Prétraitement
    signal_np = pretraiter_audio(signal_np, fs)
    
    # VAD pour isoler la zone utile
    speech_frames = detecter_parole(signal_np, fs, seuil_energie=0.005)
    if np.sum(speech_frames) > 20:
        window_size = int(0.025 * fs)
        hop_size = int(0.010 * fs)
        speech_indices = np.where(speech_frames)[0]
        start_frame = speech_indices[0] * hop_size
        end_frame = min(speech_indices[-1] * hop_size + window_size, len(signal_np))
        signal_np = signal_np[start_frame:end_frame]
    
    # Si le fichier est trop court, utiliser l'extraction standard
    if len(signal_np) < duration_per_frame * fs:
        return get_embedding_ameliore(audio_path, classifier, device=device, mode='vad')
    
    frame_length = int(duration_per_frame * fs)
    embeddings = []
    
    # Découpage uniforme en n_frames segments
    for i in range(n_frames):
        if n_frames == 1:
            start = (len(signal_np) - frame_length) // 2
        else:
            start = i * (len(signal_np) - frame_length) // (n_frames - 1)
        
        end = min(start + frame_length, len(signal_np))
        frame = signal_np[start:end]
        
        # Padding si nécessaire
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        
        frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).to(device)
        
        # Embedding de la frame
        with torch.no_grad():
            emb = classifier.encode_batch(frame_tensor)
            emb = emb.squeeze()
            emb = torch.nn.functional.normalize(emb, p=2, dim=0)
            embeddings.append(emb)
    
    # Moyenne des embeddings
    if embeddings:
        mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
        mean_embedding = torch.nn.functional.normalize(mean_embedding, p=2, dim=0)
        return mean_embedding
    
    # Fallback
    return get_embedding_ameliore(audio_path, classifier, device=device, mode='vad')

# 3. ANALYSE QUALITÉ AUDIO
# ============================

def analyser_qualite_audio_ameliore(audio_path, max_duration=10):
   """
    Analyse approfondie de la qualité du fichier audio.
    Calcule :
    - RMS : niveau global du signal
    - SNR : rapport signal/bruit estimé via énergie des frames
    - speech_ratio : proportion de frames contenant de la parole
    - duration : durée utile après prétraitement
    - quality_score : score global pondéré (durée, SNR, RMS, parole)
    """
    signal, fs = torchaudio.load(audio_path)
    
    # Conversion stéréo → mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    # Resampling en 16 kHz
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
        fs = 16000
    
    signal_np = signal.squeeze().numpy()
    total_duration = len(signal_np) / fs
    
    # Prétraitement pour l'analyse
    signal_np = pretraiter_audio(signal_np, fs)
    
    # Détection de parole VAD
    speech_frames = detecter_parole(signal_np, fs, seuil_energie=0.005)
    speech_ratio = np.sum(speech_frames) / len(speech_frames) if len(speech_frames) > 0 else 0
    
    # Troncature pour l'analyse (max 10 secondes)
    max_samples = max_duration * fs
    if len(signal_np) > max_samples:
        start = (len(signal_np) - max_samples) // 2
        signal_np = signal_np[start:start + max_samples]
    
    duration = len(signal_np) / fs
    
    # RMS du signal
    rms = np.sqrt(np.mean(signal_np**2))

# Calcul du SNR via énergie des frames
# =====================================    
   
    window_size = int(0.025 * fs)
    hop_size = int(0.010 * fs)
    
    if len(signal_np) > window_size:
        energy = np.array([
            np.sum(signal_np[i:i+window_size]**2) 
            for i in range(0, len(signal_np)-window_size, hop_size)
        ])
        
        if len(energy) > 0:
            threshold = 0.1 * np.max(energy)
            speech_frames_energy = energy > threshold
            silence_frames = ~speech_frames_energy
            
            if np.sum(speech_frames_energy) > 0 and np.sum(silence_frames) > 0:
                speech_energy = np.mean(energy[speech_frames_energy])
                silence_energy = np.mean(energy[silence_frames])
                # SNR en dB
                if silence_energy > 1e-10:
                    snr = 10 * np.log10(speech_energy / silence_energy)
                else:
                    snr = 30 # Cas extrême : silence quasi nul
            else:
                snr = 0
        else:
            snr = 0
    else:
        snr = 0
    
    # Scores normalisés (sigmoïdes / gaussiennes)
    # ============================================
    duration_score = np.exp(-((duration - 3.5)**2) / 5)
    snr_score = 1 / (1 + np.exp(-(snr - 12) / 4))
    rms_score = min(1, rms * 60)
    speech_score = min(1, speech_ratio * 2.5)
    
    # Score global pondéré
    quality_score = (duration_score * 0.25 + 
                    snr_score * 0.30 + 
                    rms_score * 0.20 + 
                    speech_score * 0.25)
    
    # Bonus pour les fichiers complets
    if 'complet' in audio_path.stem and quality_score > 0.3:
        quality_score = min(quality_score * 1.1, 1.0)
    
    return {
        'rms': rms,
        'snr': snr,
        'duration': duration,
        'total_duration': total_duration,
        'speech_ratio': speech_ratio,
        'quality_score': quality_score,
        'duration_score': duration_score,
        'snr_score': snr_score,
        'rms_score': rms_score,
        'speech_score': speech_score
    }


# ============================================
# 2. DÉTECTION DES DISCUSSIONS ET LOCUTEURS
# ============================================

def detecter_discussions_locuteurs(fichiers):
   """
    Analyse les noms de fichiers pour extraire :
    - l'ID de la discussion (numéro en début de nom)
    - le type de locuteur (Requerant, OP-SDIS1, Intervenant, autre)
    Retourne une structure :
    discussions[discussion_id][type_locuteur] = liste de fichiers
    """
    discussions = defaultdict(lambda: defaultdict(list))
    
    for f in fichiers:
        nom = f.stem
        
        # Extraction de l'ID de discussion (ex : "1234_Requerant_...")
        match = re.match(r'^(\d+)', nom)
        if match:
            discussion = match.group(1)
            
            # Détection du type de locuteur via le nom du fichier
            if 'Requerant' in nom:
                locuteur_type = 'Requerant'
            elif 'OP-SDIS1' in nom:
                locuteur_type = 'OP-SDIS1'
            elif 'Intervenant' in nom:
                locuteur_type = 'Intervenant'
            else:
                locuteur_type = 'autre'
            
            discussions[discussion][locuteur_type].append(f)
        else:
            print(f"⚠️  Impossible d'identifier la discussion pour: {nom}")
    
    return discussions


# =======================================================
# 3. SÉLECTION DE LA RÉFÉRENCE DE LA MEILLEURE RÉFÉRENCE
# =======================================================

def selectionner_reference_optimisee(corpus_dir, discussion_id, locuteur_type, 
                                     classifier, device='cpu', extraction_mode='vad'):
    """
    Sélectionne automatiquement le meilleur fichier de référence pour un locuteur donné.
    Critères :
    - qualité acoustique (quality_score)
    - similarité au centroïde des embeddings
    - durée minimale, parole suffisante
    """
    corpus_dir = Path(corpus_dir)
    
    # Recherche des fichiers correspondant au locuteur
    fichiers_type = []
    for wav_file in corpus_dir.glob("*.wav"):
        if wav_file.stem.startswith(discussion_id) and locuteur_type in wav_file.stem:
            fichiers_type.append(wav_file)
    
    if not fichiers_type:
        raise ValueError(f"Aucun fichier '{locuteur_type}' trouvé pour la discussion {discussion_id}")
    
    print(f"\n🔍 Sélection de la référence - Discussion: {discussion_id}")
    print(f"   Type: {locuteur_type}")
    print(f"   Mode d'extraction: {extraction_mode}")
    print(f"   {len(fichiers_type)} fichiers trouvés")
    print("=" * 70)
    
    candidats = []
    
    # Analyse de chaque fichier candidat
    # ===================================
    
    for audio_path in fichiers_type:
        qualite = analyser_qualite_audio_ameliore(audio_path)
        
        # Extraction de l'embedding selon le mode choisi
        if extraction_mode == 'multi_frames':
            embedding = get_embedding_multi_frames(audio_path, classifier, device=device)
        else:
            embedding = get_embedding_ameliore(audio_path, classifier, device=device, mode=extraction_mode)
        
        # Filtrage moins strict
        est_valide = True
        if qualite['total_duration'] < 1.0:
            est_valide = False
        elif qualite['speech_ratio'] < 0.1:
            est_valide = False
        elif qualite['quality_score'] < 0.1:
            est_valide = False
        
        # Affichage diagnostic
        if not est_valide:
            print(f"   ⚠️ {audio_path.name} (qualité: {qualite['quality_score']:.2f}, parole: {qualite['speech_ratio']:.2f}, durée: {qualite['total_duration']:.1f}s)")
        else:
            print(f"   ✓ {audio_path.name}")
        
        candidats.append({
            'path': audio_path,
            'name': audio_path.stem,
            'embedding': embedding,
            'qualite': qualite,
            'taille_kb': audio_path.stat().st_size / 1024,
            'est_complet': 'complet' in audio_path.stem,
            'est_valide': est_valide
        })
    
    # On garde uniquement les fichiers valides
    candidats_valides = [c for c in candidats if c['est_valide']]
    
    if not candidats_valides:
        print("\n⚠️  Aucun fichier valide trouvé! Utilisation de tous les fichiers.")
        candidats_valides = candidats
    
    print(f"\n   {len(candidats_valides)} fichiers valides sur {len(candidats)}")
    
    # Calcul du centroïde des embeddings
    # ===================================
   # Calcul du meilleur candidat en croisant la similarité au centroïde et la qualité
    quality_scores = torch.tensor([c['qualite']['quality_score'] for c in candidats_valides], device=device)
    
    embs = torch.stack([c['embedding'] for c in candidats_valides])
    centroid = torch.mean(embs, dim=0)
    centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
    # Similarité de chaque embedding au centroïde
    similarity_scores = torch.mv(embs, centroid)
    
    # Normalisation des scores
    quality_norm = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min() + 1e-10)
    similarity_norm = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min() + 1e-10)
    
    # Pondération : plus de poids sur la qualité
    poids_qualite = 0.6
    poids_similarite = 0.4
    
    combined_scores = poids_similarite * similarity_norm + poids_qualite * quality_norm
    
    # Sélection du meilleur candidat
    best_idx = torch.argmax(combined_scores).item()
    meilleur_candidat = candidats_valides[best_idx]
    
    # Affichage du TOP 10
    # =====================
    print(f"\n📊 Top 10 des candidats:")
    print("-" * 120)
    print(f"{'#':<3} {'Fichier':<35} {'Qualité':>8} {'Similarité':>10} {'Combiné':>10} {'Durée':>8} {'Parole':>8}")
    print("-" * 120)
    
    sorted_indices = torch.argsort(combined_scores, descending=True)
    for i, idx in enumerate(sorted_indices[:10]):
        c = candidats_valides[idx]
        complet_marker = " (c)" if c['est_complet'] else ""
        print(f"{i+1:<3} {c['name'][:35]:<35}{complet_marker} "
              f"{quality_norm[idx].item():>8.3f} "
              f"{similarity_norm[idx].item():>10.3f} "
              f"{combined_scores[idx].item():>10.3f} "
              f"{c['qualite']['duration']:>8.1f}s "
              f"{c['qualite']['speech_ratio']:>8.2f}")
    
    print("\n" + "=" * 70)
    print(f"🏆 MEILLEURE RÉFÉRENCE: {meilleur_candidat['name']}")
    print(f"   Qualité: {quality_norm[best_idx].item():.3f}")
    print(f"   Similarité: {similarity_norm[best_idx].item():.3f}")
    print(f"   Score combiné: {combined_scores[best_idx].item():.3f}")
    print(f"   Durée: {meilleur_candidat['qualite']['duration']:.1f}s")
    print(f"   Parole: {meilleur_candidat['qualite']['speech_ratio']:.2f}")
    if meilleur_candidat['est_complet']:
        print("   ℹ️  Fichier complet")
    print("=" * 70)
    
    return {
        'nom': meilleur_candidat['name'],
        'embedding': meilleur_candidat['embedding'],
        'score_qualite': quality_norm[best_idx].item(),
        'score_similarite': similarity_norm[best_idx].item(),
        'score_combine': combined_scores[best_idx].item(),
        'chemin': meilleur_candidat['path'],
        'discussion_id': discussion_id,
        'type_locuteur': locuteur_type,
        'est_complet': meilleur_candidat['est_complet'],
        'qualite': meilleur_candidat['qualite']
    }


# ============================================
# 4. CLASSIFICATION AVEC EXCLUSION DES INVALIDES
# ============================================

def classifier_discussion(corpus_dir, reference_info, classifier, 
                          seuil=None, device='cpu', extraction_mode='vad',
                          exclure_invalides=True):
    """
    Classifie tous les fichiers d'une discussion par rapport à une référence.
    Étapes :
    - Analyse qualité de chaque fichier
    - Exclusion des fichiers invalides (durée, parole, qualité)
    - Extraction embedding
    - Calcul du score de similarité (produit scalaire)
    - Détermination de la décision via un seuil adaptatif
    - Génération de statistiques globales et par type
    """
    corpus_dir = Path(corpus_dir)
    discussion_id = reference_info['discussion_id']
    
    print(f"\n🔍 CLASSIFICATION - Discussion {discussion_id}")
    print(f"   Référence: {reference_info['nom']} ({reference_info['type_locuteur']})")
    print(f"   Mode d'extraction: {extraction_mode}")
    print(f"   Exclure fichiers invalides: {'OUI' if exclure_invalides else 'NON'}")
    print("=" * 70)
    
    results = []
    scores = []
    stats_invalides = defaultdict(int)
    stats_invalides_par_type = defaultdict(int)
    
    # Parcours de tous les fichiers de la discussion
    # ================================================
    for wav_file in corpus_dir.glob("*.wav"):
		# On ne garde que les fichiers de la discussion
        if not wav_file.stem.startswith(discussion_id):
            continue
        # On ignore le fichier de référence    
        if wav_file.stem == reference_info['nom']:
            continue
        
        # Analyser la qualité pour vérifier la validité
        qualite = analyser_qualite_audio_ameliore(wav_file)
        
        # Déterminer le type de fichier
        if 'Requerant' in wav_file.stem:
            type_fichier = 'Requerant'
        elif 'OP-SDIS1' in wav_file.stem:
            type_fichier = 'OP-SDIS1'
        elif 'Intervenant' in wav_file.stem:
            type_fichier = 'Intervenant'
        else:
            type_fichier = 'autre'
        
        # Vérification de validité du fichier
        # =====================================
        est_valide = True
        raison_invalide = []
        
        if qualite['total_duration'] < 1.0:
            est_valide = False
            raison_invalide.append(f"durée {qualite['total_duration']:.1f}s < 1.0s")
        if qualite['speech_ratio'] < 0.1:
            est_valide = False
            raison_invalide.append(f"parole {qualite['speech_ratio']:.2f} < 0.1")
        if qualite['quality_score'] < 0.1:
            est_valide = False
            raison_invalide.append(f"qualité {qualite['quality_score']:.2f} < 0.1")
        
        # Si le fichier est invalide et qu'on les exclut, on le saute
        if exclure_invalides and not est_valide:
            stats_invalides['total'] += 1
            stats_invalides_par_type[type_fichier] += 1
            continue
        
        # Extraction et classification pour les fichiers valides
        if extraction_mode == 'multi_frames':
            embedding = get_embedding_multi_frames(wav_file, classifier, device=device)
        else:
            embedding = get_embedding_ameliore(wav_file, classifier, device=device, mode=extraction_mode)
        
        # Score de similarité (produit scalaire)
        score = torch.dot(embedding, reference_info['embedding']).item()
        scores.append(score)
        
        # Stockage du résultat
        results.append({
            'fichier': wav_file.name,
            'discussion_id': discussion_id,
            'type_fichier': type_fichier,
            'score': score,
            'embedding': embedding.cpu(),
            'taille_kb': wav_file.stat().st_size / 1024,
            'est_valide': est_valide,
            'qualite': qualite
        })
    
    # Afficher les statistiques des fichiers exclus
    if exclure_invalides and stats_invalides['total'] > 0:
        print("\n" + "-" * 70)
        print(f"📊 FICHIERS EXCLUS ({stats_invalides['total']} fichiers invalides):")
        for type_f, count in sorted(stats_invalides_par_type.items()):
            print(f"   - {type_f}: {count} fichiers")
        print("-" * 70)
    
    # Calcul dynamique du seuil en fonction de la dispersion des scores (Écart-type)
    if seuil is None:
        scores_meme_type = [r['score'] for r in results 
                           if r['type_fichier'] == reference_info['type_locuteur']]
        
        if scores_meme_type and len(scores_meme_type) > 5:
            moyenne = np.mean(scores_meme_type)
            ecart_type = np.std(scores_meme_type)
            
            # Seuil = moyenne - 0.5 × écart-type
            seuil = max(0.45, moyenne - 0.5 * ecart_type)
            print(f"\n   Seuil adaptatif: {seuil:.3f} (moyenne: {moyenne:.3f}, écart-type: {ecart_type:.3f})")
        else:
            seuil = 0.50
            print(f"\n   Seuil par défaut: {seuil:.3f}")
    
    # Appliquer la classification
    for r in results:
        r['decision'] = 'MÊME LOCUTEUR' if r['score'] >= seuil else 'LOCUTEUR DIFFÉRENT'
    
    # Tri par score décroissant
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Affichage des résultats
    # ==========================
    print(f"\n📊 Résultats de la classification ({len(results)} fichiers valides testés):")
    print("-" * 110)
    print(f"{'#':<3} {'Fichier':<50} {'Type':>12} {'Score':>10} {'Décision':>20}")
    print("-" * 110)
    
    # Affichage étendu : 60 entrées affichées pour faciliter la revue du corpus global
    for i, r in enumerate(results[:20]): # remplacer 20 par 60
        print(f"{i+1:<3} {r['fichier'][:50]:<50} {r['type_fichier']:>12} {r['score']:>10.4f} {r['decision']:>20}")
    
    if len(results) > 20: # remplacer 20 par 60
        print(f"... et {len(results)-20} autres fichiers") # remplacer 20 par 60
    
    # Statistiques par type (UNIQUEMENT sur les fichiers valides)
    type_stats = defaultdict(lambda: {'same': 0, 'diff': 0, 'total': 0})
    for r in results:
        type_stats[r['type_fichier']]['total'] += 1
        if r['decision'] == 'MÊME LOCUTEUR':
            type_stats[r['type_fichier']]['same'] += 1
        else:
            type_stats[r['type_fichier']]['diff'] += 1
    
    print("\n" + "=" * 70)
    print("📈 STATISTIQUES PAR TYPE DE LOCUTEUR")
    print("=" * 70)
    print(f"{'Type':<15} {'Total valides':>15} {'Même locuteur':>15} {'Différent':>15} {'% Même':>10}")
    print("-" * 70)
    
    for type_f, data in sorted(type_stats.items()):
        pct_same = data['same'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"{type_f:<15} {data['total']:>15} {data['same']:>15} {data['diff']:>15} {pct_same:>9.1f}%")
    
    # Statistiques globales
    nb_same = sum(1 for r in results if r['decision'] == 'MÊME LOCUTEUR')
    nb_diff = len(results) - nb_same
    
    print("\n" + "=" * 70)
    print("📊 STATISTIQUES GLOBALES")
    print("=" * 70)
    print(f"Total fichiers valides testés: {len(results)}")
    print(f"Même locuteur: {nb_same} ({nb_same/len(results)*100:.1f}%)")
    print(f"Locuteur différent: {nb_diff} ({nb_diff/len(results)*100:.1f}%)")
    if exclure_invalides and stats_invalides['total'] > 0:
        print(f"Fichiers exclus (invalides): {stats_invalides['total']}")
    print(f"Seuil utilisé: {seuil:.3f}")
    print("=" * 70)
    
    return results, type_stats, seuil


# ============================================
# 5. VISUALISATION
# ============================================

def visualiser_resultats(results, reference_info, seuil, discussion_id):
    """
    Génère un tableau de bord graphique :
    - histogramme des scores
    - scatter plot par type
    - boxplot par type
    - heatmap de similarité (top 20 embeddings)
    """
    
    # Conversion CPU si nécessaire
    for r in results:
        if hasattr(r['embedding'], 'cpu'):
            r['embedding'] = r['embedding'].cpu()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Histogramme des scores
    # ===========================
    scores = [r['score'] for r in results]
    
    axes[0, 0].hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(x=seuil, color='red', linestyle='--', linewidth=2, label=f'Seuil = {seuil:.3f}')
    axes[0, 0].set_xlabel('Score de similarité')
    axes[0, 0].set_ylabel('Nombre de fichiers')
    axes[0, 0].set_title(f'Distribution - Discussion {discussion_id}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter plot par type
    # ==========================
    
    types = ['Requerant', 'OP-SDIS1', 'Intervenant', 'autre']
    colors_type = {'Requerant': 'blue', 'OP-SDIS1': 'green', 'Intervenant': 'orange', 'autre': 'gray'}
    
    for type_f in types:
        type_scores = [r['score'] for r in results if r['type_fichier'] == type_f]
        if type_scores:
            x = np.random.normal(0, 0.1, len(type_scores)) + list(types).index(type_f)
            axes[0, 1].scatter(x, type_scores, alpha=0.6, s=30, 
                             color=colors_type.get(type_f, 'gray'), label=type_f)
    
    axes[0, 1].axhline(y=seuil, color='red', linestyle='--', linewidth=2, label=f'Seuil = {seuil:.3f}')
    axes[0, 1].set_xlabel('Type de fichier')
    axes[0, 1].set_ylabel('Score de similarité')
    axes[0, 1].set_title('Scores par type')
    axes[0, 1].set_xticks(range(len(types)))
    axes[0, 1].set_xticklabels(types)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Boxplot par type
    # =======================
    
    type_scores_list = []
    type_labels = []
    for type_f in types:
        type_scores = [r['score'] for r in results if r['type_fichier'] == type_f]
        if type_scores:
            type_scores_list.append(type_scores)
            type_labels.append(type_f)
    
    if type_scores_list:
        bp = axes[0, 2].boxplot(type_scores_list, tick_labels=type_labels, patch_artist=True)
        axes[0, 2].axhline(y=seuil, color='red', linestyle='--', linewidth=2, label=f'Seuil = {seuil:.3f}')
        axes[0, 2].set_xlabel('Type')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Distribution des scores par type')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Heatmap de similarité (top 20 embeddings)
    # ================================================
    , an
    if len(results) > 3:
        top_n = min(20, len(results))
        top_results = results[:top_n]
        top_embeddings = torch.stack([r['embedding'] for r in top_results])
        similarity_matrix = torch.mm(top_embeddings, top_embeddings.T).numpy()
        
        im = axes[1, 0].imshow(similarity_matrix, cmap='coolwarm', aspect='auto', 
                               vmin=0, vmax=1)
        axes[1, 0].set_xticks(range(len(top_results)))
        axes[1, 0].set_yticks(range(len(top_results)))
        axes[1, 0].set_xticklabels([r['fichier'][:15] for r in top_results], rotation=45, ha='right', fontsize=8)
        axes[1, 0].set_yticklabels([r['fichier'][:15] for r in top_results], fontsize=8)
        axes[1, 0].set_title('Matrice de similarité (top 20)')
        plt.colorbar(im, ax=axes[1, 0])
    
    # 5. Statistiques par type (bar chart)
    types_present = [t for t in types if any(r['type_fichier'] == t for r in results)]
    type_counts = []
    type_same = []
    
    for t in types_present:
        type_results = [r for r in results if r['type_fichier'] == t]
        type_counts.append(len(type_results))
        type_same.append(sum(1 for r in type_results if r['decision'] == 'MÊME LOCUTEUR'))
    
    if type_counts:
        x = range(len(types_present))
        axes[1, 1].bar(x, type_counts, alpha=0.7, color='blue', label='Total')
        axes[1, 1].bar(x, type_same, alpha=0.7, color='green', label='Même locuteur')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(types_present)
        axes[1, 1].set_xlabel('Type')
        axes[1, 1].set_ylabel('Nombre')
        axes[1, 1].set_title('Distribution par type')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Scores vs taille
    tailles = [r['taille_kb'] for r in results]
    scores = [r['score'] for r in results]
    colors = ['green' if r['decision'] == 'MÊME LOCUTEUR' else 'red' for r in results]
    
    axes[1, 2].scatter(tailles, scores, c=colors, alpha=0.6, s=30)
    axes[1, 2].axhline(y=seuil, color='red', linestyle='--', linewidth=2, label=f'Seuil = {seuil:.3f}')
    axes[1, 2].set_xlabel('Taille du fichier (KB)')
    axes[1, 2].set_ylabel('Score de similarité')
    axes[1, 2].set_title('Score vs Taille de fichier')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# 6. SAUVEGARDE
# ============================================

def sauvegarder_resultats(results, reference_info, type_stats, seuil, 
                         output_dir="resultats"):
   """
Les fonctions de sauvegarde permettent de conserver les résultats de la
classification pour analyse ultérieure, audit ou visualisation.

Éléments sauvegardés :
- embeddings : signatures vocales normalisées (format .pt)
- résultats de classification : tableau CSV contenant :
    * nom du fichier
    * score de similarité
    * décision (même locuteur / différent)
    * type de locuteur
    * métriques de qualité audio
- référence sélectionnée : embedding + métadonnées
- logs textuels : résumé de la discussion, seuil utilisé, statistiques

Intérêt :
- Permet de rejouer la classification sans recalculer les embeddings
- Facilite l’analyse comparative entre discussions
- Permet de constituer un corpus propre pour l’entraînement futur
"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    discussion_id = reference_info['discussion_id']
    type_ref = reference_info['type_locuteur']
    
    # Sauvegarde des tenseurs pour une éventuelle ré-exploitation
    corpus_embeddings = {}
    for r in results:
        corpus_embeddings[r['fichier']] = r['embedding']
    corpus_embeddings[reference_info['nom']] = reference_info['embedding']
    
    torch.save(corpus_embeddings, output_dir / f"embeddings_{discussion_id}_{type_ref}_{timestamp}.pt")
    print(f"✅ Embeddings sauvegardés")
    
    # Rapport structure tabulaire
    csv_path = output_dir / f"resultats_{discussion_id}_{type_ref}_{timestamp}.csv"
    df = pd.DataFrame([{
        'Fichier': r['fichier'],
        'Type': r['type_fichier'],
        'Score': r['score'],
        'Decision': r['decision']
    } for r in results])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ Résultats sauvegardés: {csv_path.name}")
    
    # (Sauvegardes Textes et Référence PyTorch ...)
    stats_path = output_dir / f"stats_{discussion_id}_{type_ref}_{timestamp}.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"RAPPORT D'ANALYSE - Discussion {discussion_id}\n")
        f.write(f"Référence: {reference_info['type_locuteur']} - {reference_info['nom']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Fichier de référence: {reference_info['nom']}\n")
        f.write(f"Score qualité: {reference_info['score_qualite']:.3f}\n")
        f.write(f"Score similarité: {reference_info['score_similarite']:.3f}\n")
        f.write(f"Score combiné: {reference_info['score_combine']:.3f}\n")
        f.write(f"Seuil utilisé: {seuil:.3f}\n\n")
        
        f.write("STATISTIQUES PAR TYPE\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Type':<15} {'Total':>8} {'Même':>10} {'Différent':>12} {'% Même':>10}\n")
        f.write("-" * 50 + "\n")
        
        for type_f, data in sorted(type_stats.items()):
            pct_same = data['same'] / data['total'] * 100 if data['total'] > 0 else 0
            f.write(f"{type_f:<15} {data['total']:>8} {data['same']:>10} {data['diff']:>12} {pct_same:>9.1f}%\n")
        
        nb_same = sum(1 for r in results if r['decision'] == 'MÊME LOCUTEUR')
        f.write(f"\nTotal Même locuteur: {nb_same}/{len(results)} ({nb_same/len(results)*100:.1f}%)\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ Rapport sauvegardé: {stats_path.name}")
    
    # 4. Référence
    torch.save({
        'nom': reference_info['nom'],
        'embedding': reference_info['embedding'],
        'discussion_id': reference_info['discussion_id'],
        'type_locuteur': reference_info['type_locuteur'],
        'score_qualite': reference_info['score_qualite'],
        'score_similarite': reference_info['score_similarite']
    }, output_dir / f"reference_{discussion_id}_{type_ref}_{timestamp}.pt")
    print(f"✅ Référence sauvegardée")


# ============================================
# 7. ANALYSE COMPLÈTE
# ============================================

def analyser_discussion_complete(corpus_dir, discussion_id, type_reference, 
                                 classifier, device='cpu', extraction_mode='vad',
                                 seuil=None, visualiser=True, exclure_invalides=True):
    """Analyse complète d'une discussion"""
    print(f"\n{'#'*70}")
    print(f"ANALYSE COMPLÈTE - Discussion {discussion_id}")
    print(f"Référence: {type_reference}")
    print(f"Mode d'extraction: {extraction_mode}")
    print(f"Exclure fichiers invalides: {'OUI' if exclure_invalides else 'NON'}")
    print(f"{'#'*70}")
    
    # 1. Sélection de la référence
    reference = selectionner_reference_optimisee(
        corpus_dir, discussion_id, type_reference, 
        classifier, device, extraction_mode
    )
    
    # 2. Classification avec exclusion des invalides
    results, type_stats, seuil_utilise = classifier_discussion(
        corpus_dir, reference, classifier, seuil, device, extraction_mode,
        exclure_invalides=exclure_invalides
    )
    
    # 3. Visualisation
    if visualiser:
        visualiser_resultats(results, reference, seuil_utilise, discussion_id)
    
    # 4. Sauvegarde
    sauvegarder_resultats(results, reference, type_stats, seuil_utilise)
    
    return {
        'reference': reference,
        'results': results,
        'type_stats': type_stats,
        'seuil': seuil_utilise
    }


# ============================================
# 8. FONCTIONS INTER-DISCUSSIONS (INTÉGRÉES DE LA V1)
# ============================================

def classifier_corpus_entier(corpus_dir, reference_info, classifier, 
                             seuil=None, device='cpu', extraction_mode='vad',
                             exclure_invalides=True):
								 
	"""
Cette partie étend la classification au-delà d'une seule discussion.

Objectif :
    Tester une référence (ex : un opérateur SDIS) sur l'ensemble du corpus
    afin de :
    - détecter les locuteurs récurrents
    - repérer les confusions entre discussions
    - vérifier la cohérence des annotations
    - identifier les fichiers mal nommés ou mal catégorisés

Fonctionnement :
    1. Sélection d'une référence dans une discussion donnée
    2. Parcours de tous les fichiers du corpus
    3. Extraction embedding + score de similarité
    4. Application du seuil (fixe ou adaptatif)
    5. Génération :
        - d'un tableau global des scores
        - d'une heatmap inter-discussion
        - d'un histogramme global
        - de statistiques par discussion

Cas d’usage SDIS / NexSIS :
    - Vérifier si un opérateur apparaît dans plusieurs appels
    - Détecter les erreurs de segmentation ou de nommage
    - Identifier les requérants récurrents
    - Auditer la qualité du corpus avant annotation manuelle
"""
    corpus_dir = Path(corpus_dir)
    discussion_ref = reference_info['discussion_id']
    
    print(f"\n🔍 CLASSIFICATION SUR TOUT LE CORPUS")
    print(f"   Référence: {reference_info['nom']} ({reference_info['type_locuteur']})")
    print(f"   Discussion source: {discussion_ref}")
    print(f"   Mode d'extraction: {extraction_mode}")
    print(f"   Exclure fichiers invalides: {'OUI' if exclure_invalides else 'NON'}")
    print("=" * 70)
    
    results = []
    scores = []
    stats_invalides = defaultdict(int)
    stats_invalides_par_type = defaultdict(int)
    stats_par_discussion = defaultdict(lambda: {'same': 0, 'diff': 0, 'total': 0})
    
    for wav_file in corpus_dir.glob("*.wav"):
        if wav_file.stem == reference_info['nom']:
            continue
        
        qualite = analyser_qualite_audio_ameliore(wav_file)
        
        discussion_id = re.match(r'^(\d+)', wav_file.stem).group(1) if re.match(r'^(\d+)', wav_file.stem) else 'unknown'
        
        if 'Requerant' in wav_file.stem:
            type_fichier = 'Requerant'
        elif 'OP-SDIS1' in wav_file.stem:
            type_fichier = 'OP-SDIS1'
        elif 'Intervenant' in wav_file.stem:
            type_fichier = 'Intervenant'
        else:
            type_fichier = 'autre'
        
        est_valide = True
        if qualite['total_duration'] < 1.0:
            est_valide = False
        elif qualite['speech_ratio'] < 0.1:
            est_valide = False
        elif qualite['quality_score'] < 0.1:
            est_valide = False
        
        if exclure_invalides and not est_valide:
            stats_invalides['total'] += 1
            stats_invalides_par_type[f"{discussion_id}_{type_fichier}"] += 1
            continue
        
        if extraction_mode == 'multi_frames':
            embedding = get_embedding_multi_frames(wav_file, classifier, device=device)
        else:
            embedding = get_embedding_ameliore(wav_file, classifier, device=device, mode=extraction_mode)
        
        score = torch.dot(embedding, reference_info['embedding']).item()
        scores.append(score)
        
        results.append({
            'fichier': wav_file.name,
            'discussion_id': discussion_id,
            'type_fichier': type_fichier,
            'score': score,
            'embedding': embedding.cpu(),
            'taille_kb': wav_file.stat().st_size / 1024,
            'est_valide': est_valide,
            'est_meme_discussion': (discussion_id == discussion_ref),
            'qualite': qualite
        })
    
    if exclure_invalides and stats_invalides['total'] > 0:
        print("\n" + "-" * 70)
        print(f"📊 FICHIERS EXCLUS ({stats_invalides['total']} fichiers invalides):")
        for key, count in sorted(stats_invalides_par_type.items()):
            print(f"   - {key}: {count} fichiers")
        print("-" * 70)
    
    if seuil is None:
        scores_meme_type = [r['score'] for r in results 
                           if r['type_fichier'] == reference_info['type_locuteur']]
        
        if scores_meme_type and len(scores_meme_type) > 5:
            moyenne = np.mean(scores_meme_type)
            ecart_type = np.std(scores_meme_type)
            seuil = max(0.45, moyenne - 0.5 * ecart_type)
            print(f"\n   Seuil adaptatif: {seuil:.3f} (moyenne: {moyenne:.3f}, écart-type: {ecart_type:.3f})")
        else:
            seuil = 0.50
            print(f"\n   Seuil par défaut: {seuil:.3f}")
    
    for r in results:
        r['decision'] = 'MÊME LOCUTEUR' if r['score'] >= seuil else 'LOCUTEUR DIFFÉRENT'
        stats_par_discussion[r['discussion_id']]['total'] += 1
        if r['decision'] == 'MÊME LOCUTEUR':
            stats_par_discussion[r['discussion_id']]['same'] += 1
        else:
            stats_par_discussion[r['discussion_id']]['diff'] += 1
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n📊 Résultats de la classification inter-discussions ({len(results)} fichiers valides testés):")
    print("-" * 120)
    print(f"{'#':<3} {'Fichier':<45} {'Discussion':>10} {'Type':>12} {'Score':>10} {'Décision':>20}")
    print("-" * 120)
    
    for i, r in enumerate(results[:60]): # on peut changer le nombre de fichiers à l'affichage 25 ou 60
        disc_marker = " ⭐" if r['discussion_id'] == discussion_ref else "  "
        print(f"{i+1:<3} {r['fichier'][:45]:<45} {r['discussion_id']:>10}{disc_marker} {r['type_fichier']:>12} {r['score']:>10.4f} {r['decision']:>20}")
    
    if len(results) > 60:# on peut changer le nombre de fichiers à l'affichage 25 ou 60
        print(f"... et {len(results)-60} autres fichiers")# on peut changer le nombre de fichiers à l'affichage 25 ou 60
    
    stats_par_type = defaultdict(lambda: {'same': 0, 'diff': 0, 'total': 0})
    for r in results:
        key = f"{r['discussion_id']} {r['type_fichier']}"
        stats_par_type[key]['total'] += 1
        if r['decision'] == 'MÊME LOCUTEUR':
            stats_par_type[key]['same'] += 1
        else:
            stats_par_type[key]['diff'] += 1
    
    return results, stats_par_type, stats_par_discussion, seuil


def visualiser_inter_discussions(results, reference_info, seuil, discussion_ref):
    for r in results:
        if hasattr(r['embedding'], 'cpu'):
            r['embedding'] = r['embedding'].cpu()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    scores = [r['score'] for r in results]
    axes[0, 0].hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(x=seuil, color='red', linestyle='--', linewidth=2, label=f'Seuil = {seuil:.3f}')
    axes[0, 0].set_title(f'Distribution inter-discussions\nRéf: {reference_info["nom"]}')
    axes[0, 0].legend()
    
    discussions = sorted(set(r['discussion_id'] for r in results))
    disc_scores = []
    disc_labels = []
    
    for disc in discussions:
        disc_scores_list = [r['score'] for r in results if r['discussion_id'] == disc]
        if disc_scores_list:
            disc_scores.append(disc_scores_list)
            disc_labels.append(f"{disc} {'⭐' if disc == discussion_ref else ''}")
    
    if disc_scores:
        axes[0, 1].boxplot(disc_scores, tick_labels=disc_labels, patch_artist=True)
        axes[0, 1].axhline(y=seuil, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Scores par discussion')
    
    types = ['Requerant', 'OP-SDIS1', 'Intervenant', 'autre']
    colors_type = {'Requerant': 'blue', 'OP-SDIS1': 'green', 'Intervenant': 'orange', 'autre': 'gray'}
    
    for type_f in types:
        type_scores = [r['score'] for r in results if r['type_fichier'] == type_f]
        if type_scores:
            x = np.random.normal(0, 0.1, len(type_scores)) + list(types).index(type_f)
            axes[0, 2].scatter(x, type_scores, alpha=0.6, s=30, color=colors_type.get(type_f, 'gray'), label=type_f)
    
    axes[0, 2].axhline(y=seuil, color='red', linestyle='--', linewidth=2)
    axes[0, 2].set_xticks(range(len(types)))
    axes[0, 2].set_xticklabels(types)
    axes[0, 2].set_title('Scores par type')
    axes[0, 2].legend()
    
    if len(results) > 3:
        top_n = min(25, len(results))
        top_results = results[:top_n]
        top_embeddings = torch.stack([r['embedding'] for r in top_results])
        similarity_matrix = torch.mm(top_embeddings, top_embeddings.T).numpy()
        labels = [f"{r['discussion_id']}:{r['fichier'][:10]}" for r in top_results]
        im = axes[1, 0].imshow(similarity_matrix, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_xticks(range(len(top_results)))
        axes[1, 0].set_yticks(range(len(top_results)))
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        axes[1, 0].set_yticklabels(labels, fontsize=7)
        axes[1, 0].set_title('Matrice de similarité (top 25)')
        plt.colorbar(im, ax=axes[1, 0])
    
    discussions_present = sorted(set(r['discussion_id'] for r in results))
    disc_counts, disc_same = [], []
    for disc in discussions_present:
        disc_results = [r for r in results if r['discussion_id'] == disc]
        disc_counts.append(len(disc_results))
        disc_same.append(sum(1 for r in disc_results if r['decision'] == 'MÊME LOCUTEUR'))
    
    if disc_counts:
        x = range(len(discussions_present))
        axes[1, 1].bar(x, disc_counts, alpha=0.7, color='blue', label='Total')
        axes[1, 1].bar(x, disc_same, alpha=0.7, color='green', label='Même locuteur')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f"{d} {'⭐' if d == discussion_ref else ''}" for d in discussions_present])
        axes[1, 1].set_title('Distribution par discussion')
        axes[1, 1].legend()
    
    if disc_counts:
        pct_same = [disc_same[i] / disc_counts[i] * 100 if disc_counts[i] > 0 else 0 for i in range(len(disc_counts))]
        axes[1, 2].bar(x, pct_same, alpha=0.7, color='green')
        axes[1, 2].axhline(y=seuil*100, color='red', linestyle='--', linewidth=2)
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels([f"{d} {'⭐' if d == discussion_ref else ''}" for d in discussions_present])
        axes[1, 2].set_title('Taux de reconnaissance (%)')
        axes[1, 2].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()


def sauvegarder_inter_discussions(results, reference_info, stats_par_type, 
                                  stats_par_discussion, seuil, output_dir="resultats"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    discussion_ref = reference_info['discussion_id']
    type_ref = reference_info['type_locuteur']
    
    corpus_embeddings = {r['fichier']: r['embedding'] for r in results}
    corpus_embeddings[reference_info['nom']] = reference_info['embedding']
    torch.save(corpus_embeddings, output_dir / f"embeddings_interdisc_{discussion_ref}_{type_ref}_{timestamp}.pt")
    
    df = pd.DataFrame([{
        'Fichier': r['fichier'], 'Discussion': r['discussion_id'],
        'Type': r['type_fichier'], 'Score': r['score'], 'Decision': r['decision']
    } for r in results])
    df.to_csv(output_dir / f"resultats_interdisc_{discussion_ref}_{type_ref}_{timestamp}.csv", index=False)
    print(f"✅ Résultats et embeddings sauvegardés dans '{output_dir}'.")


# ============================================
# 9. MAIN (MIS À JOUR AVEC CHOIX DU PÉRIMÈTRE)
# ============================================

def main():
	
	"""
    Le main orchestre l'ensemble du pipeline de classification :

    Étapes :
        1. Chargement du modèle ECAPA-TDNN (SpeechBrain)
        2. Sélection de la discussion source
        3. Sélection du type de locuteur (Requerant / OP-SDIS1 / Intervenant)
        4. Sélection automatique de la meilleure référence :
            - qualité audio
            - similarité au centroïde
            - durée utile
            - densité de parole
        5. Classification intra-discussion :
            - extraction embeddings
            - calcul des scores
            - seuil adaptatif
            - statistiques par type
            - visualisation
        6. Option : classification inter-discussion
        7. Sauvegarde des résultats :
            - CSV
            - embeddings
            - logs
            - figures

     Intérêt :
        - Pipeline complet, robuste et automatisé
        - Adapté aux corpus téléphoniques bruités (SDIS / NexSIS)
        - Permet d’auditer la cohérence des locuteurs
        - Permet de nettoyer un corpus avant annotation
        - Permet de constituer des références fiables pour l’entraînement

     Notes :
        - Le pipeline est modulaire : chaque étape peut être activée/désactivée
        - Le seuil adaptatif améliore la robustesse sur les corpus hétérogènes
        - Les visualisations facilitent l’analyse qualitative
    """
    print("=" * 70)
    print("SYSTÈME DE VÉRIFICATION - EXTRACTION AMÉLIORÉE V2 (AVEC MULTI-DISCUSSIONS)")
    print("=" * 70)
    
    print("\n📥 Chargement du modèle SpeechBrain...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   ℹ️  Utilisation de {device}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Chargement modèle
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
    
    corpus_dir = "mes_audios/test"
    fichiers = list(Path(corpus_dir).glob("*.wav"))
    discussions = detecter_discussions_locuteurs(fichiers)
    
    print(f"\n📋 Discussions et locuteurs détectés:")
    for discussion_id, locuteurs in discussions.items():
        print(f"   - Discussion {discussion_id} : {sum(len(f) for f in locuteurs.values())} fichiers")
    
    # 1. Choix de la discussion
    print("\n" + "=" * 70)
    print("1. Choisissez la discussion SOURCE (pour la référence):")
    for i, disc_id in enumerate(discussions.keys(), 1):
        print(f"   {i}. Discussion {disc_id}")
    
    choix_disc = input("\nVotre choix (numéro): ").strip()
    try:
        discussion_source = list(discussions.keys())[int(choix_disc) - 1]
    except:
        discussion_source = list(discussions.keys())[0]
        print(f"Choix invalide. Utilisation de la discussion {discussion_source}.")
    
    # 2. Choix du type de référence
    print("\n" + "=" * 70)
    print("2. Choisissez le type de locuteur pour la référence:")
    print("   1. Requerant\n   2. OP-SDIS1\n   3. Intervenant")
    type_map = {'1': 'Requerant', '2': 'OP-SDIS1', '3': 'Intervenant'}
    type_reference = type_map.get(input("\nVotre choix (1-3): ").strip(), 'Requerant')
    
    if type_reference not in discussions.get(discussion_source, {}):
        print(f"\n⚠️ Type '{type_reference}' introuvable dans {discussion_source}.")
        type_reference = list(discussions[discussion_source].keys())[0]
        print(f"   -> Remplacement automatique par : {type_reference}")
    
    # 3. Choix du PÉRIMÈTRE D'ANALYSE
    print("\n" + "=" * 70)
    print("3. Choisissez le périmètre d'analyse:")
    print("   1. Analyser UNIQUEMENT la discussion choisie (Intra-discussion)")
    print("   2. Analyser sur TOUT le corpus (Inter-discussions)")
    mode_scope = input("\nVotre choix (1-2): ").strip()

    # 4. Mode d'extraction
    print("\n" + "=" * 70)
    print("4. Choisissez le mode d'extraction:")
    print("   1. VAD (détection de parole) - RECOMMANDÉ")
    print("   2. Top-K (meilleures frames)")
    print("   3. Multi-frames (moyenne de plusieurs extractions)")
    print("   4. Standard")
    mode_map = {'1': 'vad', '2': 'top_k', '3': 'multi_frames', '4': 'standard'}
    extraction_mode = mode_map.get(input("\nVotre choix (1-4): ").strip(), 'vad')

    # 5. Seuil
    print("\n" + "=" * 70)
    print("5. Seuil de décision:")
    print("   1. Adaptatif (RECOMMANDÉ)\n   2. Manuel")
    if input("\nVotre choix (1-2): ").strip() == '2':
        try:
            seuil = float(input("\nEntrez le seuil (ex: 0.45): ").strip())
        except:
            seuil = None
    else:
        seuil = None

    # 6. Exclusion
    exclure_invalides = input("\nExclure les fichiers invalides ? 1. OUI, 2. NON : ").strip() != '2'

    # ============================================
    # EXÉCUTION
    # ============================================
    print("\n" + "=" * 70)
    print("🚀 DÉBUT DE L'ANALYSE")
    print("=" * 70)

    if mode_scope == '2':
        reference = selectionner_reference_optimisee(
            corpus_dir, discussion_source, type_reference, classifier, device, extraction_mode
        )
        results, stats_type, stats_disc, seuil_utilise = classifier_corpus_entier(
            corpus_dir, reference, classifier, seuil, device, extraction_mode, exclure_invalides
        )
        visualiser_inter_discussions(results, reference, seuil_utilise, discussion_source)
        sauvegarder_inter_discussions(results, reference, stats_type, stats_disc, seuil_utilise)
    else:
        analyser_discussion_complete(
            corpus_dir, discussion_source, type_reference, classifier, device, 
            extraction_mode, seuil, visualiser=True, exclure_invalides=exclure_invalides
        )

    print("\n" + "=" * 70)
    print("✅ PROCESSUS TERMINÉ")
    print("=" * 70)

if __name__ == "__main__":
    main()
