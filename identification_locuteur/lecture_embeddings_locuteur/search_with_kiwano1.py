#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torch.nn.functional import cosine_similarity

"""
Recherche interactive d'un locuteur dans un corpus d'embeddings.

Ce script permet de :
- Charger un fichier d'embeddings (format .pt ou .pkl Kiwano) comme référence
- Comparer cette référence à un corpus d'embeddings
- Utiliser différentes méthodes de scoring : cosinus simple, S-Norm, AS-Norm, AD-Norm
- Afficher et sauvegarder les résultats avec un seuil personnalisable

Dépendances :
- torch
- base.py (à placer dans le même dossier pour le support des fichiers .pkl)
"""

# --------------------------------------------------------------------
#  INTÉGRATION DE base.py (Kiwano embedding I/O)
#  Si base.py est présent dans le même dossier, on l'importe.
#  Sinon, on utilise torch.load pour les fichiers .pt (fallback).
# --------------------------------------------------------------------
try:
    from base import load_embeddings as kiwano_load_embeddings
    HAS_KIWANO_BASE = True
except ImportError:
    HAS_KIWANO_BASE = False
    print("INFO: base.py non trouvé. Seuls les fichiers .pt seront supportés.")


def load_embeddings(file_path):
        """
    Charge un fichier d'embeddings.
    - .pt  : torch.load (dictionnaire PyTorch)
    - .pkl : utilise le module base.py (format Kiwano) si disponible

    Args:
        file_path (str): Chemin vers le fichier

    Returns:
        dict: Dictionnaire {nom_fichier: tenseur_embedding}
    """
    file_path = str(file_path)
    if file_path.endswith('.pkl') and HAS_KIWANO_BASE:
        # Format Kiwano : spécification pkl:...
        return kiwano_load_embeddings(f"pkl:{file_path}")
    else:
        # Fallback : .pt ou tout autre fichier PyTorch
        data = torch.load(file_path, map_location='cpu')
        if not isinstance(data, dict):
            raise ValueError("Le fichier doit contenir un dictionnaire {nom: tenseur}")
        return data


# ===================================================================
#  FONCTIONS DE SCORING (inspirées des utilitaires Kiwano)
#  Ces fonctions implémentent les méthodes de comparaison de locuteurs
#  couramment utilisées dans la vérification du locuteur.
# ===================================================================

cos = torch.nn.CosineSimilarity(dim=0)

def compute_raw_cosine(xvector_enrollment, xvector_test):
    """
    Calcule la similarité cosinus brute entre deux embeddings.
    C'est la méthode la plus simple et la plus rapide.

    Args:
        xvector_enrollment (Tensor): Embedding de référence
        xvector_test (Tensor): Embedding à tester

    Returns:
        float: Score de similarité entre -1 et 1
    """
    return cos(xvector_enrollment, xvector_test).item()

def compute_v(xvector, impostors):
    """
    Calcule la similarité cosinus entre un xvector et tous les impostors.
    Utilisé pour les méthodes de normalisation.

    Args:
        xvector (Tensor): Embedding à comparer
        impostors (dict): Dictionnaire d'embeddings des imposteurs

    Returns:
        Tensor: Scores de similarité avec tous les imposteurs
    """
    return torch.stack([cos(xvector, impostors[imp]) for imp in impostors])

def compute_score_snorm(xvector_enrollment, xvector_test, enrollment_name, test_name, impostors, mean_std_cache):
    """
    Calcule le score S-Norm (Symetric Normalization).

    S-Norm = 0.5 * ((score - mean_enrollment) / std_enrollment + (score - mean_test) / std_test)

    Principe : normalise le score brut en utilisant la moyenne et l'écart-type
    des scores entre le locuteur cible et un ensemble d'imposteurs.

    Args:
        xvector_enrollment (Tensor): Embedding de référence
        xvector_test (Tensor): Embedding à tester
        enrollment_name (str): Nom de l'embedding de référence
        test_name (str): Nom de l'embedding testé
        impostors (dict): Dictionnaire des embeddings imposteurs
        mean_std_cache (dict): Cache des moyennes et écarts-types calculés

    Returns:
        float: Score normalisé (typiquement centré autour de 0)
    """
    if enrollment_name not in mean_std_cache:
        ve = compute_v(xvector_enrollment, impostors)
        mean_std_cache[enrollment_name] = (torch.mean(ve), torch.std(ve))
    if test_name not in mean_std_cache:
        vt = compute_v(xvector_test, impostors)
        mean_std_cache[test_name] = (torch.mean(vt), torch.std(vt))
    
    score = cos(xvector_enrollment, xvector_test)
    mean_e, std_e = mean_std_cache[enrollment_name]
    mean_t, std_t = mean_std_cache[test_name]
    
    return (((score - mean_e) / (std_e)) + ((score - mean_t) / (std_t))) * 0.5

def compute_score_asnorm(xvector_enrollment, xvector_test, enrollment_name, test_name, impostors, k, mean_std_cache):
    """
    Calcule le score AS-Norm (Adaptive S-Norm).

    Principe : sélectionne les k meilleurs imposteurs (les plus proches)
    pour le calcul de la moyenne et de l'écart-type. Plus robuste que S-Norm
    car elle ignore les imposteurs très éloignés.

    Args:
        xvector_enrollment (Tensor): Embedding de référence
        xvector_test (Tensor): Embedding à tester
        enrollment_name (str): Nom de l'embedding de référence
        test_name (str): Nom de l'embedding testé
        impostors (dict): Dictionnaire des embeddings imposteurs
        k (int): Nombre d'imposteurs à sélectionner
        mean_std_cache (dict): Cache des moyennes et écarts-types

    Returns:
        float: Score normalisé (typiquement centré autour de 0)
    """
    if enrollment_name not in mean_std_cache:
        ve = compute_v(xvector_enrollment, impostors)
        top_vals, _ = torch.topk(ve, k)
        mean_std_cache[enrollment_name] = (torch.mean(top_vals), torch.std(top_vals))
    if test_name not in mean_std_cache:
        vt = compute_v(xvector_test, impostors)
        top_vals, _ = torch.topk(vt, k)
        mean_std_cache[test_name] = (torch.mean(top_vals), torch.std(top_vals))
    
    score = cos(xvector_enrollment, xvector_test)
    mean_e, std_e = mean_std_cache[enrollment_name]
    mean_t, std_t = mean_std_cache[test_name]
    
    return (((score - mean_e) / (std_e)) + ((score - mean_t) / (std_t))) * 0.5

def compute_score_adnorm(xvector_enrollment, xvector_test, enrollment_name, test_name, impostors, k, mean_cache):
    """
    Calcule le score AD-Norm (Adaptive Domain Normalization).

    Principe : soustrait la moyenne des k meilleurs imposteurs directement
    au niveau des embeddings avant de calculer la similarité. Cette méthode
    est particulièrement efficace pour corriger les biais de domaine.

    Args:
        xvector_enrollment (Tensor): Embedding de référence
        xvector_test (Tensor): Embedding à tester
        enrollment_name (str): Nom de l'embedding de référence
        test_name (str): Nom de l'embedding testé
        impostors (dict): Dictionnaire des embeddings imposteurs
        k (int): Nombre d'imposteurs à sélectionner
        mean_cache (dict): Cache des moyennes adaptatives calculées

    Returns:
        float: Score de similarité (typiquement entre -1 et 1)
    """
    impostor_keys = list(impostors.keys())
    
    if enrollment_name not in mean_cache:
        ve = compute_v(xvector_enrollment, impostors)
        _, top_idx = torch.topk(ve, k)
        mean_cache[enrollment_name] = torch.mean(
            torch.stack([impostors[impostor_keys[i.item()]] for i in top_idx])
        )
    if test_name not in mean_cache:
        vt = compute_v(xvector_test, impostors)
        _, top_idx = torch.topk(vt, k)
        mean_cache[test_name] = torch.mean(
            torch.stack([impostors[impostor_keys[i.item()]] for i in top_idx])
        )
    
    return cos(
        xvector_enrollment - mean_cache[enrollment_name],
        xvector_test - mean_cache[test_name]
    ).item()

# ===================================================================
#  FONCTIONS D'INTERFACE UTILISATEUR
# ===================================================================

def list_pt_or_pkl_files():
    """Liste tous les fichiers .pt et .pkl du répertoire courant."""
    files = [f for f in os.listdir('.') if f.endswith(('.pt', '.pkl'))]
    return files

def choose_file(prompt, files):
    """Affiche une liste de fichiers et demande à l'utilisateur d'en choisir un."""
    print(f"\n{prompt}")
    for i, f in enumerate(files, 1):
        print(f"  {i:2d} : {f}")
    while True:
        try:
            choice = int(input("Entrez le numéro du fichier : "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            print(f"Numéro invalide. Choisissez entre 1 et {len(files)}.")
        except ValueError:
            print("Entrez un nombre valide.")

def list_ref_keys(ref_data):
    """Affiche les clés (noms de fichiers) contenues dans le fichier de référence."""
    print("\nFichiers de référence disponibles dans ce fichier :")
    keys = list(ref_data.keys())
    for i, k in enumerate(keys, 1):
        print(f"  {i:2d} : {k}")
    return keys

def choose_ref_key(ref_data):
	"""Demande à l'utilisateur de choisir une clé de référence."""
    keys = list_ref_keys(ref_data)
    while True:
        try:
            choice = int(input("Entrez le numéro du fichier à utiliser comme référence : "))
            if 1 <= choice <= len(keys):
                return keys[choice - 1]
            print(f"Numéro invalide. Choisissez entre 1 et {len(keys)}.")
        except ValueError:
            print("Entrez un nombre valide.")

def get_scoring_method():
    """
    Demande à l'utilisateur de choisir la méthode de scoring.

    Retourne:
        int: 1=cosinus, 2=S-Norm, 3=AS-Norm, 4=AD-Norm
    """
    print("\nMéthodes de scoring disponibles :")
    print("  1 : Cosinus simple (raw cosine)")
    print("  2 : S-Norm (normalisation symétrique)")
    print("  3 : AS-Norm (Adaptive S-Norm)")
    print("  4 : AD-Norm (Adaptive Domain Normalization)")
    while True:
        try:
            choice = int(input("Entrez le numéro de la méthode : "))
            if 1 <= choice <= 4:
                return choice
            print("Choix invalide. Choisissez entre 1 et 4.")
        except ValueError:
            print("Entrez un nombre valide.")

def get_threshold(default=0.75):
	    """
    Demande le seuil de similarité.

    Note : Pour le cosinus, le seuil est typiquement entre 0.7 et 0.85.
           Pour S-Norm et AS-Norm, les scores sont normalisés autour de 0,
           un seuil de 1.5 à 2.5 est souvent approprié.
           Pour AD-Norm, les scores sont comparables au cosinus.
    """
    while True:
        val = input(f"Seuil de similarité (défaut: {default}) : ").strip()
        if not val:
            return default
        try:
            th = float(val)
            return th  # on accepte tout nombre réel (pas de contrainte)
        except ValueError:
            print("Entrez un nombre valide.")

def get_k(default=100):
    """
    Demande la taille de la cohorte pour AS-Norm et AD-Norm.
    K détermine le nombre d'imposteurs les plus proches utilisés pour la normalisation.
    Une valeur typique est 50-150.
    """
    while True:
        val = input(f"Taille de la cohorte K (défaut: {default}) : ").strip()
        if not val:
            return default
        try:
            k = int(val)
            if k > 0:
                return k
            print("K doit être un nombre positif.")
        except ValueError:
            print("Entrez un nombre valide.")

def get_n_results(default=60):
	"""Demande le nombre de meilleurs résultats à afficher."""
    val = input(f"Nombre de résultats à afficher (défaut: {default}) : ").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default

def get_output_file(default="resultats_kiwano.txt"):
	"""Demande le nom du fichier de sortie."""
    out = input(f"Nom du fichier de sortie (défaut: {default}) : ").strip()
    return out if out else default

# ===================================================================
#  COMPARAISON ET LABELLISATION
# ===================================================================

def compare_and_label(corpus_data, template, ref_name, method, k, threshold):
    """
    Compare le template à tous les embeddings du corpus.

    Args:
        corpus_data (dict): Dictionnaire des embeddings du corpus
        template (Tensor): Embedding de référence
        ref_name (str): Nom de l'embedding de référence
        method (int): 1=cosinus, 2=S-Norm, 3=AS-Norm, 4=AD-Norm
        k (int): Taille de cohorte (pour AS-Norm et AD-Norm)
        threshold (float): Seuil pour le label

    Returns:
        list: Liste de tuples (nom_fichier, score, label)
    """
    results = []
    
    if method == 1:
        # Cosinus simple : comparaison directe
        for name, emb in corpus_data.items():
            sim = compute_raw_cosine(template, emb)
            label = "MÊME LOCUTEUR" if sim >= threshold else "LOCUTEUR DIFFÉRENT"
            results.append((name, sim, label))
    
    else:
        # Méthodes avec normalisation (S-Norm, AS-Norm, AD-Norm)
        # On utilise tous les autres embeddings du corpus comme ensemble d'imposteurs
        impostor_keys = [k for k in corpus_data.keys() if k != ref_name]
        if not impostor_keys:
            print("Erreur : pas assez d'embeddings dans le corpus pour former un ensemble d'impostors.")
            return []
        
        impostors = {k: corpus_data[k] for k in impostor_keys}
        mean_std_cache = {}
        mean_cache = {}
        
        for name, emb in corpus_data.items():
            if method == 2:
                sim = compute_score_snorm(template, emb, ref_name, name, impostors, mean_std_cache)
            elif method == 3:
                sim = compute_score_asnorm(template, emb, ref_name, name, impostors, k, mean_std_cache)
            elif method == 4:
                sim = compute_score_adnorm(template, emb, ref_name, name, impostors, k, mean_cache)
            else:
                continue
            
            # Utilisation du seuil saisi par l'utilisateur pour le label
            label = "MÊME LOCUTEUR" if sim > threshold else "LOCUTEUR DIFFÉRENT"
            results.append((name, sim, label))
    
    # Tri par score décroissant
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ===================================================================
#  AFFICHAGE ET SAUVEGARDE
# ===================================================================

def save_and_print(results, output_file, ref_name, method_name, k, threshold, ref_file, corpus_file, n_results=60):
	    """
    Affiche les résultats dans le terminal et les sauvegarde dans un fichier.

    Args:
        results (list): Liste de tuples (nom, score, label)
        output_file (str): Chemin du fichier de sortie
        ref_name (str): Nom de la référence utilisée
        method_name (str): Nom de la méthode de scoring
        k (int): Taille de cohorte (si applicable)
        threshold (float): Seuil utilisé
        ref_file (str): Chemin du fichier de référence
        corpus_file (str): Chemin du fichier corpus
        n_results (int): Nombre de résultats à afficher dans le terminal
    """
    print("\n" + "=" * 80)
    print("RÉSULTATS DE LA RECHERCHE")
    print(f"Fichier de référence   : {ref_file}")
    print(f"Fichier corpus         : {corpus_file}")
    print(f"Référence utilisée     : {ref_name}")
    print(f"Méthode de scoring     : {method_name}")
    if method_name in ["AS-Norm", "AD-Norm"]:
        print(f"Taille de cohorte K    : {k}")
    print(f"Seuil                   : {threshold}")
    print(f"Nombre de segments      : {len(results)}")
    print("=" * 80)
    print(f"{'NOM DU FICHIER':<50} {'SCORE':>12} {'JUGEMENT':<20}")
    print("-" * 80)

    # Affichage limité aux n premiers résultats
    for name, score, label in results[:n_results]:
        print(f"{name:<50} {score:>12.4f} {label:<20}")

    if len(results) > n_results:
        print(f"... et {len(results) - n_results} autres résultats (voir fichier complet).")

    print("=" * 80)

    # Écriture complète dans le fichier
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Fichier de référence : {ref_file}\n")
        f.write(f"Fichier corpus       : {corpus_file}\n")
        f.write(f"Référence utilisée   : {ref_name}\n")
        f.write(f"Méthode de scoring   : {method_name}\n")
        if method_name in ["AS-Norm", "AD-Norm"]:
            f.write(f"Taille de cohorte K  : {k}\n")
        f.write(f"Seuil                : {threshold}\n")
        f.write(f"Nombre de segments   : {len(results)}\n\n")
        f.write(f"{'NOM DU FICHIER':<50} {'SCORE':>12} {'JUGEMENT':<20}\n")
        f.write("-" * 80 + "\n")
        for name, score, label in results:
            f.write(f"{name:<50} {score:>12.4f} {label:<20}\n")
        same = sum(1 for _, _, lab in results if lab == "MÊME LOCUTEUR")
        diff = len(results) - same
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Résumé : {same} 'MÊME LOCUTEUR', {diff} 'LOCUTEUR DIFFÉRENT'.\n")

    print(f"\nRésultats enregistrés dans : {output_file}")
    print(f"Résumé : {same} segments 'MÊME LOCUTEUR', {diff} 'LOCUTEUR DIFFÉRENT'.")

# ===================================================================
#  MAIN INTERACTIF
# ===================================================================

def main():
	"""
    Fonction principale du script.

    Étapes :
    1. Liste et choix du fichier de référence (.pt ou .pkl)
    2. Choix du fichier de référence parmi les clés du dictionnaire
    3. Choix du fichier corpus
    4. Choix de la méthode de scoring
    5. Paramétrage (K pour AS-Norm/AD-Norm, seuil, nombre de résultats)
    6. Chargement des données
    7. Comparaison
    8. Affichage et sauvegarde
    """
    print("\n=== RECHERCHE INTERACTIVE DE LOCUTEUR AVEC KIWANO ===\n")

    # 1. Lister les fichiers .pt et .pkl
    files = list_pt_or_pkl_files()
    if not files:
        print("Aucun fichier .pt ou .pkl trouvé dans le répertoire courant.")
        return
    ref_file = choose_file("Choisissez le fichier d'embeddings de référence :", files)

    # 2. Charger et choisir la clé de référence
    ref_data = load_embeddings(ref_file)
    ref_name = choose_ref_key(ref_data)

    # 3. Choisir le fichier corpus
    corpus_choices = [f for f in files if f != ref_file]
    if not corpus_choices:
        print("Aucun autre fichier pour le corpus.")
        return
    corpus_file = choose_file("Choisissez le fichier du corpus :", corpus_choices)

    # 4. Choisir la méthode de scoring
    method_choice = get_scoring_method()
    method_names = {1: "Cosinus simple", 2: "S-Norm", 3: "AS-Norm", 4: "AD-Norm"}
    method_name = method_names[method_choice]

    # 5. Si AS-Norm ou AD-Norm, demander K
    k = None
    if method_choice in [3, 4]:
        k = get_k(100)

    # 6. Seuil
    # Valeur par défaut adaptée à la méthode
    # - Cosinus : 0.75 (score entre -1 et 1)
    # - S-Norm / AS-Norm : 2.0 (score normalisé centré autour de 0)
    # - AD-Norm : 0.75 (score comparable au cosinus)
    if method_choice == 1:
        default_threshold = 0.75
    elif method_choice == 4:
        default_threshold = 0.75
    else:  # S-Norm ou AS-Norm
        default_threshold = 2.0
    threshold = get_threshold(default_threshold)

    # 7. Nombre de résultats à afficher
    n_results = get_n_results(60)

    # 8. Fichier de sortie
    output_file = get_output_file("resultats_kiwano.txt")

    # 9. Charger le corpus
    corpus_data = load_embeddings(corpus_file)
    print(f"Corpus chargé : {len(corpus_data)} embeddings.")

    # 10. Extraire le template (embedding de référence)
    template = ref_data[ref_name]
    print("Template extrait.")

    # 11. Comparer
    print("Comparaison en cours...")
    results = compare_and_label(corpus_data, template, ref_name, method_choice, k, threshold)

    if not results:
        print("Aucun résultat.")
        return

    # 12. Afficher et sauvegarder
    save_and_print(results, output_file, ref_name, method_name, k, threshold, ref_file, corpus_file, n_results)

if __name__ == "__main__":
    main()
