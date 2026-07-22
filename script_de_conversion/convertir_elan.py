#!/usr/bin/env python3
# Script pour convertir l'export ELAN vers le format Audacity
# Ce script lit un fichier texte exporté depuis ELAN, extrait les annotations et les regroupe par intervenant (locuteur).
# Chaque intervenant aura son propre fichier de sortie au format Audacity, contenant les segments temporels correspondants.
# Prérequis : Python 3 et le fichier d'entrée doit être au format texte (exporté depuis ELAN) avec des colon
# nes séparées par des tabulations.
# Usage : python convertir_elan.py mon_fichier.txt
# Version : 1 fichier par intervenant

import sys
import os

# Demander le nom du fichier
print("=== Convertisseur ELAN -> Audacity (1 fichier par intervenant) ===")
print("")
fichier_entree = input("Nom du fichier ELAN a convertir (ex: mon_fichier.txt) : ")

# Vérifier que le fichier existe
if not os.path.exists(fichier_entree):
    print(f"ERREUR: Le fichier '{fichier_entree}' n'existe pas.")
    print(f"Assurez-vous qu'il est dans le dossier: {os.getcwd()}")
    sys.exit(1)

# Extraire le nom de base (sans l'extension)
nom_base = os.path.splitext(fichier_entree)[0]

# Lire et grouper par intervenant
intervenants = {}  # dictionnaire: {"OP-SDIS1": [lignes], "Requerant1": [lignes]}

with open(fichier_entree, 'r', encoding='utf-8') as entree:
    lignes = entree.readlines()

for ligne in lignes:
    ligne = ligne.strip()
    if not ligne:
        continue
    
    colonnes = ligne.split('\t')
    
    # Format attendu: [piste] [annotateur] [debut] [fin] [texte]
    if len(colonnes) >= 5:
        debut = colonnes[2]
        fin = colonnes[3]
        locuteur = colonnes[0]  # OP-SDIS1, Requerant1, etc.
        
        # Stocker la ligne au format Audacity (juste le nom, pas le texte)
        ligne_audacity = f"{debut}\t{fin}\t{locuteur}\n"
        
        if locuteur not in intervenants:
            intervenants[locuteur] = []
        intervenants[locuteur].append(ligne_audacity)
    else:
        print(f"Ligne ignorée (mauvais format): {ligne}")

# Créer un fichier par intervenant
print("")
for locuteur, lignes_audacity in intervenants.items():
    fichier_sortie = f"{locuteur}_audacity.txt"
    with open(fichier_sortie, 'w', encoding='utf-8') as sortie:
        sortie.writelines(lignes_audacity)
    print(f"✓ Fichier créé : {fichier_sortie} ({len(lignes_audacity)} étiquettes)")

print("")
print("=== CONVERSION TERMINEE ===")
print("")
print("=== Prochaines etapes dans Audacity ===")
print("Pour CHAQUE fichier *_audacity.txt :")
print("1. Ouvrez votre fichier audio .wav original")
print("2. Fichier > Importer > Etiquettes")
print("3. Selectionnez le fichier *_audacity.txt")
print("4. Fichier > Exporter > Exporter plusieurs...")
print("5. Separer en fichier selon: Etiquettes")
print("6. Renommer le fichier: Utilisation du nom de l'etiquette")
print("7. Exporter -> vous obtenez UN fichier .wav par intervenant")
print("")
print("Fermez et rouvrez le fichier audio original entre chaque intervenant.")
