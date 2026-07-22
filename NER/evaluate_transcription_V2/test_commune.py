import json
import os
import re

# utilitaire pour tester le gazetteer des communes de Haute-Garonne
# a partir du fichier communes-haute-garonne.geojson.json
# Objectif : vérifier que la commune "saint sulpice sur leze" est bien présente
# et que la regex fonctionne correctement pour la détecter dans une phrase
# et apprecier les différentes formes d'écriture possibles (accents, tirets, etc.)

# 1. Chargement du gazetteer (copie exacte de votre fonction)
def charger_gazetteer(chemin_fichier):
    villes = []
    if os.path.exists(chemin_fichier):
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            if chemin_fichier.endswith('.json'):
                donnees = json.load(f)
                for feature in donnees.get('features', []):
                    ville = feature.get('properties', {}).get('nom_commune', '')
                    if ville:
                        villes.append(ville.strip().lower().replace("-", " "))
        print(f"[INFO] {len(set(villes))} communes chargées.")
    return list(set(villes))

# 2. Chargement
print("="*60)
print("TEST DE CHARGEMENT DU GAZETTEER")
print("="*60)

GAZETTEER_COMMUNES_31 = charger_gazetteer("communes-haute-garonne.geojson.json")

# 3. Recherche de la commune
print(f"\nRecherche de 'saint sulpice sur leze' :")
trouvee = False

# Test 1 : Présence directe
if "saint sulpice sur leze" in GAZETTEER_COMMUNES_31:
    print("✅ Trouvée telle quelle")
    trouvee = True

# Test 2 : Recherche partielle
print("\nCommunes contenant 'sulpice' :")
for ville in GAZETTEER_COMMUNES_31:
    if "sulpice" in ville.lower():
        print(f"  → '{ville}'")
        trouvee = True

# Test 3 : Recherche dans le fichier source directement
print("\nRecherche dans le fichier source :")
with open("communes-haute-garonne.geojson.json", 'r', encoding='utf-8') as f:
    contenu = f.read()
    if "sulpice" in contenu.lower():
        # Trouver le contexte
        index = contenu.lower().find("sulpice")
        contexte = contenu[max(0,index-100):index+100]
        print(f"✅ Trouvé dans le fichier source !")
        print(f"   Contexte : ...{contexte}...")
    else:
        print(f"❌ PAS trouvé dans le fichier source !")
        print(f"   La commune n'est PAS dans communes-haute-garonne.geojson.json")

# Test 4 : Test regex
print("\nTest de la regex :")
REGEX_COMMUNES = r"\b(" + "|".join(GAZETTEER_COMMUNES_31) + r")\b"
test_phrase = "l'accident a eu lieu à saint sulpice sur leze"
match = re.search(REGEX_COMMUNES, test_phrase)
if match:
    print(f"✅ La regex a trouvé : '{match.group(0)}'")
else:
    print(f"❌ La regex n'a PAS trouvé la commune")

# Test 5 : Vérification des accents
print("\nVérification des accents :")
for ville in GAZETTEER_COMMUNES_31:
    if "leze" in ville.lower() or "lèze" in ville.lower():
        print(f"  → '{ville}'")

print("="*60)
