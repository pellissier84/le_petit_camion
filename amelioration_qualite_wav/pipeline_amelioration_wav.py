import os
import glob
import subprocess
import shutil
import logging
import numpy as np
import torch
import torchaudio
from scipy import signal as scipy_signal
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier

"""
Script optionnel 
Pipeline Batch : Amélioration et Évaluation Vectorielle de la Parole
Auteur : 
Date : Juillet 2026
Description : Nettoyage conditionnel (Demucs) validé par analyse de variance (SpeechBrain ECAPA-TDNN).
"""

# ==========================================
# 1. MOTEUR DE NETTOYAG (DEMUCS)
# ==========================================
def nettoyer_avec_demucs(chemin_audio):
    """
    Exécute Demucs en arrière-plan via subprocess.
    Isole la piste vocale (vocals) pour supprimer les bruits parasites.
    """
    print(f"      [ÉTAPE] Lancement de Demucs sur {chemin_audio}...")

    # --two-stems=vocals permet de gagner du temps de calcul en ne séparant que la voix du reste
    commande = ["demucs", "--two-stems=vocals", chemin_audio]

    try:
        subprocess.run(commande, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return None

    # Reconstitution du chemin de sortie standard de Demucs
    nom_dossier = os.path.basename(chemin_audio).rsplit('.', 1)[0]
    chemin_nettoye = os.path.join("separated", "htdemucs", nom_dossier, "vocals.wav")

    return chemin_nettoye if os.path.exists(chemin_nettoye) else None


# ==========================================
# 2. PRÉTRAITEMENT ACOUSTIQUE & VAD & SNR
# ==========================================
def pretraiter_audio(signal_np, fs):
    """
    Applique une normalisation RMS et un filtre passe-haut pour nettoyer les basses fréquences.
    """

    # Normalisation du volume pour uniformiser l'énergie avant extraction
    rms = np.sqrt(np.mean(signal_np**2))
    if rms > 1e-6:
        signal_np = signal_np / rms * 0.1

    # Filtre passe-haut (80 Hz) pour couper les bruits de manipulation et ronronnements
    try:
        b, a = scipy_signal.butter(4, 80 / (fs / 2), 'high')
        signal_np = scipy_signal.filtfilt(b, a, signal_np).copy()
    except Exception:
        pass

    return signal_np


def detecter_parole(signal_np, fs, seuil_energie=0.005):
    """
    Voice Activity Detection (VAD) basique basé sur l'énergie par tranches de 25ms.
    """
    window_size = int(0.025 * fs)
    hop_size = int(0.010 * fs)

    if len(signal_np) < window_size:
        return np.array([True])

    # Calcul de l'énergie au carré pour chaque frame
    energy = np.array([
        np.sum(signal_np[i:i + window_size]**2)
        for i in range(0, len(signal_np) - window_size, hop_size)
    ])

    if len(energy) == 0:
        return np.array([True])

    return energy > (seuil_energie * np.max(energy))


def estimer_snr(signal_np, fs, seuil_energie=0.005):
    """
    Calcule le Rapport Signal/Bruit (SNR) en séparant les frames de parole des frames de silence.
    """
    window_size = int(0.025 * fs)
    hop_size = int(0.010 * fs)

    if len(signal_np) < window_size:
        return 50.0

    energy = np.array([
        np.sum(signal_np[i:i + window_size]**2)
        for i in range(0, len(signal_np) - window_size, hop_size)
    ])

    if len(energy) == 0:
        return 50.0

    threshold = seuil_energie * np.max(energy)

    # Séparation Signal (Parole) / Bruit (Silences)
    energie_parole = energy[energy > threshold]
    energie_bruit = energy[energy <= threshold]

    moy_parole = np.mean(energie_parole) if len(energie_parole) > 0 else 0
    moy_bruit = np.mean(energie_bruit) if len(energie_bruit) > 0 else 1e-10

    if moy_bruit < 1e-10:
        moy_bruit = 1e-10

    # Formule standard du SNR en Décibels (dB)
    return 10 * np.log10(moy_parole / moy_bruit)


# ==========================================
# 3. EXTRACTION VECTORIELLE (Avec modèle injecté)
# ==========================================
def analyser_et_extraire(audio_path, classifier, label=""):
    """
    Découpe le signal en fenêtres, extrait les embeddings avec SpeechBrain, 
    et calcule la similarité cosinus par rapport au centroïde (la voix "moyenne").
   
    Retourne (similarite_moyenne, ecart_type, nb_fenetres)
    """
    print(f"      --- Analyse : {label} ---")
    
    # Chargement et forçage du taux d'échantillonnage à 16kHz (requis par ECAPA-TDNN)
    signal, fs = torchaudio.load(audio_path)
    if signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
        fs = 16000
        
    signal_np = signal.squeeze().numpy()
    signal_pretraite = pretraiter_audio(signal_np, fs)
    duree_sec = len(signal_pretraite) / fs
    speech_frames = detecter_parole(signal_pretraite, fs)
    ratio_global = np.sum(speech_frames) / len(speech_frames) if len(speech_frames) > 0 else 0
    
    # Adaptation dynamique : on réduit la fenêtre pour pouvoir analyser les fichiers très courts
    if duree_sec >= 3.0:
        window_samples, step_samples = int(3.0 * fs), int(1.0 * fs)
    else:
        window_samples, step_samples = int(0.5 * fs), int(0.25 * fs)
        
    vecteurs_valides = []
    
    # CAS 1 : Fichier extrême (plus court que la fenêtre), on extrait un vecteur unique
    if len(signal_pretraite) < window_samples:
        if ratio_global > 0.1:
            tensor_fenetre = torch.from_numpy(signal_pretraite).float().unsqueeze(0)
            embedding = classifier.encode_batch(tensor_fenetre)    
            vecteurs_valides.append(embedding.squeeze().cpu().numpy())
    # CAS 2 : Découpage par fenêtre glissante classique        
    else:
        for start in range(0, len(signal_pretraite) - window_samples + 1, step_samples):
            end = start + window_samples
            debut_frame, fin_frame = int(start / fs * 100), int(end / fs * 100)
            frames_fenetre = speech_frames[debut_frame:fin_frame]
            ratio_fenetre = np.sum(frames_fenetre) / len(frames_fenetre) if len(frames_fenetre) > 0 else 0
            
            # On ignore les fenêtres composées majoritairement de silence
            if ratio_fenetre > 0.15:
                tensor_fenetre = torch.from_numpy(signal_pretraite[start:end]).float().unsqueeze(0)
                embedding = classifier.encode_batch(tensor_fenetre)
                vecteurs_valides.append(embedding.squeeze().cpu().numpy())
            
    nb_fenetres = len(vecteurs_valides)
    if nb_fenetres == 0: return 0, 0, 0
        
    matrice_valide = np.array(vecteurs_valides)
    centroide = np.mean(matrice_valide, axis=0)
    
    # Calcul de la variance : plus elle est faible, plus la voix est pure et constante
    if nb_fenetres == 1:
        moyenne, ecart_type = 1.0, 0.0
    else:
        similarites = [1 - cosine(centroide, v) for v in matrice_valide]
        moyenne, ecart_type = np.mean(similarites), np.std(similarites)
    
    print(f"      -> {nb_fenetres} vecteurs | Variance: {ecart_type:.4f}")
    return moyenne, ecart_type, nb_fenetres

# ==========================================
# 4. BOUCLE DE TRAITEMENT PAR LOT (BATCH)
# ==========================================
if __name__ == "__main__":
	
	# --- CONFIGURATION ---
    DOSSIER_CIBLE = "." # "." désigne le dossier courant. Vous pouvez y mettre un chemin absolu.
    SEUIL_SNR_DB = 20.0 # Seuil au-dessus duquel l'audio est jugé suffisamment propre
    
    # 1. Lister tous les fichiers WAV, sauf ceux déjà estampillés "_ameliore"
    tous_les_wav = glob.glob(os.path.join(DOSSIER_CIBLE, "*.wav"))
    fichiers_a_traiter = [f for f in tous_les_wav if not f.endswith("_ameliore.wav")]
    
    if not fichiers_a_traiter:
        print("Aucun fichier WAV à traiter dans ce dossier.")
        exit(0)
        
    print("==================================================")
    print(f" INITIALISATION DU LOT : {len(fichiers_a_traiter)} FICHIERS DÉTECTÉS")
    print("==================================================")
    
    # 2. Chargement du modèle une seule fois pour tout le lot (Gain énorme de temps/RAM)
    print("Chargement du modèle d'empreinte vocale...")
    logging.getLogger("speechbrain").setLevel(logging.ERROR)
    modele_vocal = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")
    
    # Statistiques globales
    stats = {"propres": 0, "ameliores": 0, "degrades": 0, "erreurs": 0}
    
    # 3. Boucle principale : Exécution de la boucle sur le lot
    for i, fichier_brut in enumerate(fichiers_a_traiter, 1):
        print(f"\n[{i}/{len(fichiers_a_traiter)}] Traitement de : {os.path.basename(fichier_brut)}")
        
        try:
            signal, fs = torchaudio.load(fichier_brut)
            if signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True)
            
            # Évaluation rapide du SNR
            snr_estime = estimer_snr(pretraiter_audio(signal.squeeze().numpy(), fs), fs)
            print(f"      -> SNR estimé : {snr_estime:.1f} dB")
            
            # Application du Gatekeeper
            if snr_estime >= SEUIL_SNR_DB:
                print("      [DÉCISION] Audio propre. Ignoré pour éviter la distorsion.")
                stats["propres"] += 1
                continue
                
            print("      [DÉCISION] Audio bruité. Nettoyage en cours...")
            moy_brut, std_brut, nb_brut = analyser_et_extraire(fichier_brut, modele_vocal, "BRUT")
            fichier_nettoye = nettoyer_avec_demucs(fichier_brut)
            
            if fichier_nettoye:
                moy_net, std_net, nb_net = analyser_et_extraire(fichier_nettoye, modele_vocal, "NETTOYÉ")
                
                # A/B Testing : Validation mathématique du nettoyag
                if std_net < std_brut:
                    # Le nettoyage a rendu les embeddings plus stables -> On sauvegarde
                    nom_base, ext = os.path.splitext(os.path.basename(fichier_brut))
                    chemin_sauvegarde = os.path.join(DOSSIER_CIBLE, f"{nom_base}_ameliore{ext}")
                    shutil.copy2(fichier_nettoye, chemin_sauvegarde)
                    
                    # Archivage du fichier brut d'origine
                    dossier_archives = os.path.join(DOSSIER_CIBLE, "archives")
                    os.makedirs(dossier_archives, exist_ok=True)
                    shutil.move(fichier_brut, os.path.join(dossier_archives, os.path.basename(fichier_brut)))
                    
                    print(f"      [SUCCÈS] Variance réduite ! Fichier archivé.")
                    stats["ameliores"] += 1
                else:
                    print(f"      [ÉCHEC] Variance augmentée. Fichier conservé tel quel.")
                    stats["degrades"] += 1
                
                # Nettoyage des dossiers temporaires de traitement Demucs 
                try: shutil.rmtree("separated")
                except: pass
                
        except Exception as e:
            print(f"      [ERREUR CRITIQUE] Impossible de traiter {fichier_brut}: {e}")
            stats["erreurs"] += 1

    # 4. Rapport Final
    print("\n\n==================================================")
    print(" RAPPPORT DE TRAITEMENT GLOBAL")
    print("==================================================")
    print(f" Fichiers scannés    : {len(fichiers_a_traiter)}")
    print(f" Audios déjà propres : {stats['propres']}")
    print(f" Audios améliorés    : {stats['ameliores']} (originaux déplacés dans 'archives/')")
    print(f" Audios non probants : {stats['degrades']} (suppression du nettoyage)")
    print(f" Erreurs de lecture  : {stats['erreurs']}")
    print("==================================================")
