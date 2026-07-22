# Pipeline d'Amélioration Intelligente des Empreintes Vocales

Ce projet propose un pipeline automatisé de traitement audio (Batch Processing) conçu pour préparer et 
nettoyer des enregistrements WAV en vue d'une tâche de diarisation ou d'identification du locuteur.

Contrairement à un script de nettoyage classique, cet outil intègre un **mécanisme de décision scientifique (Gatekeeper)** : 
il s'assure mathématiquement que le nettoyage par IA n'altère pas le timbre vocal (distorsion) avant de valider 
la sauvegarde du fichier.


Ce pipeline se comporte exactement comme un système de production automatisé.

Voici les points clés qui valident son travail par un test :

    On peut maîtrisé le seuil : on peut ajusté la variable SEUIL_SNR_DB 
    ( 20.0 dB a la place de 15.0 dB), Lors d'essai en modifiant le seuil, puisque le fichier à 19.7 dB a été traité, 
    tandis que celui à 20.6 dB a été ignoré. 
    
    L'A/B Testing automatisé fonctionne : Sur le fichier _016.wav, le script a détecté une amélioration de la variance 
    (de 0.0031 à 0.0030). Demucs a donc réussi à retirer un bruit parasite sans abîmer le timbre de la voix. 
    Le script a pris la bonne décision en archivant l'original et en conservant la version améliorée.

    L'optimisation des ressources : Le modèle d'intelligence artificielle ne s'est chargé qu'une seule fois au tout début 
    (Chargement du modèle d'empreinte vocale...), ce qui fait gagner un temps précieux sur le traitement en boucle.
On dispose d'un outil d'ingénierie audio complet, mathématiquement justifiable et totalement automatisé.


Architecture et Logique

Le script traite un dossier complet d'enregistrements audio en suivant cette logique de validation A/B :

	1. **Gatekeeper Acoustique (SNR) :** Le script calcule d'abord le Rapport Signal/Bruit (SNR). 
			Si l'audio est déjà propre (ex: > 20 dB), le nettoyage lourd est ignoré pour éviter le sur-traitement.
	2. **Extraction de Référence :** Si l'audio est bruité, le script extrait les vecteurs vocaux (embeddings) 
			sur le fichier brut grâce au modèle **ECAPA-TDNN (SpeechBrain)** et calcule la variance de 
			ces vecteurs (éparpillement dû au bruit).
	3. **Nettoyage Profond (Demucs) :** Le fichier passe dans le réseau de neurones Demucs pour isoler la piste vocale.
	4. **Évaluation Comparative :** Le script extrait les vecteurs vocaux sur la version nettoyée. 
	   - **Si la variance diminue :** Le bruit a été retiré avec succès sans altérer la voix. 
			Le fichier nettoyé remplace l'original, et l'original est archivé.
	   - **Si la variance augmente :** L'IA a déformé le timbre de la voix (artefacts). 
			Le nettoyage est rejeté et l'original est conservé.

##  Prérequis et Installation

Le script s'exécute sous environnement Linux et requiert un GPU (recommandé) ou un CPU.


# Installation des dépendances requises
pip install torch torchaudio scipy numpy speechbrain demucs

Utilisation

    Placez le script pipeline_ameliore_wav.py dans le dossier contenant vos fichiers .wav.

    Ajustez la variable SEUIL_SNR_DB au début du bloc d'exécution selon l'acoustique de votre projet (généralement entre 15.0 et 20.0).

    Lancez le script :
    python pipeline_ameliore_wav.py
    
Structure des données après exécution

    *_ameliore.wav : Les fichiers validés scientifiquement par l'algorithme.

    archives/ : Dossier généré automatiquement contenant les fichiers bruts originaux ayant bénéficié d'une amélioration.
    
exemple de sortie : 

			==================================================
			 INITIALISATION DU LOT : 4 FICHIERS DÉTECTÉS
			==================================================
			Chargement du modèle d'empreinte vocale...
			Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
			Could not parse CUDA device string 'cuda': not enough values to unpack (expected 2, got 1). Falling back to device 0.

			[1/4] Traitement de : 1826_OP-SDIS1-11.wav
				  -> SNR estimé : 22.3 dB
				  [DÉCISION] Audio propre. Ignoré.

			[2/4] Traitement de : audio-1775031826.41778_OP-SDIS1_016.wav
				  -> SNR estimé : 19.7 dB
				  [DÉCISION] Audio bruité. Nettoyage en cours...
				  --- Analyse : BRUT ---
				  -> 2 vecteurs | Variance: 0.0031
				  [ÉTAPE] Lancement de Demucs sur ./audio-1775031826.41778_OP-SDIS1_016.wav...
				  --- Analyse : NETTOYÉ ---
				  -> 2 vecteurs | Variance: 0.0030
				  [SUCCÈS] Variance réduite ! Fichier archivé.

			[3/4] Traitement de : audio-1775031826.41778_OP-SDIS1_003.wav
				  -> SNR estimé : 22.2 dB
				  [DÉCISION] Audio propre. Ignoré.

			[4/4] Traitement de : 1826_OP-SDIS1-30.wav
				  -> SNR estimé : 20.6 dB
				  [DÉCISION] Audio propre. Ignoré.


			==================================================
			 RAPPPORT DE TRAITEMENT GLOBAL
			==================================================
			 Fichiers scannés    : 4
			 Audios déjà propres : 3
			 Audios améliorés    : 1 (originaux déplacés dans 'archives/')
			 Audios non probants : 0 (suppression du nettoyage)
			 Erreurs de lecture  : 0
			==================================================
