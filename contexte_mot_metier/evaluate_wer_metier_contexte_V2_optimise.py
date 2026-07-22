#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évaluation des mots métier avec analyse contextuelle
Nom du fichier : evaluate_wer_metier_contexte_V2_optimise.py

DESCRIPTION ARCHITECTURALE :
Ce script compare une transcription automatique (WhisperX) à une référence humaine (ELAN)
pour évaluer la reconnaissance d'un vocabulaire métier spécifique (SDIS/CISU).
Il utilise une approche hybride à 3 niveaux pour l'extraction :
1. Règles strictes (Expressions Régulières - Regex)
2. Dictionnaire métier (spaCy EntityRuler via le fichier JSONL)
3. Intelligence Artificielle générique (spaCy NER)
Il intègre ensuite une analyse du contexte (mots environnants) pour confirmer
ou infirmer la pertinence des termes détectés.

Utilisation :
python evaluate_wer_metier_contexte_V2_optimise.py --eaf "audio-1775031826.41778.eaf" --json "audio-1775031826.41778.json"
"""

import json
import re
import sys
import argparse
import os
from collections import defaultdict
from difflib import SequenceMatcher
import unicodedata

# ============================================================
# IMPORTS SPÉCIFIQUES
# ============================================================
# On vérifie si les bibliothèques cruciales sont installées avant de lancer le script
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    print("[ERREUR] spaCy n'est pas installé. Exécutez : pip install spacy && python -m spacy download fr_core_news_sm")
    HAS_SPACY = False
    sys.exit(1)

try:
    import pympi  # Bibliothèque pour lire les fichiers .eaf générés par le logiciel ELAN
    HAS_PYMPI = True
except ImportError:
    print("[ERREUR] pympi n'est pas installé. Exécutez : pip install pympi-ling")
    HAS_PYMPI = False
    sys.exit(1)

# ============================================================
# CHARGEMENT DE SPACY
# ============================================================
print("Chargement du modèle d'Intelligence Artificielle (spaCy)... patientez.")
try:
	# Utilisation du modèle 'Large' (lg) pour une meilleure compréhension du vocabulaire médical
    nlp = spacy.load("fr_core_news_lg") # charger sm petit modéle ou lg grand modèle
    print("✓ spaCy chargé avec succès")
except:
    print("[ERREUR] Modèle fr_core_news_sm non trouvé. Exécutez : python -m spacy download fr_core_news_sm")
    sys.exit(1)

# ============================================================
# CHARGEMENT DU GAZETTEER (Communes 31)
# ============================================================
def charger_gazetteer(chemin_fichier):
	"""
    Charge une liste de noms de villes à partir d'un fichier JSON ou CSV.
    Permet au script de reconnaître dynamiquement les lieux d'intervention.
    """
    villes = []
    if os.path.exists(chemin_fichier):
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            if chemin_fichier.endswith('.json'):
                donnees = json.load(f)
                for feature in donnees.get('features', []):
                    ville = feature.get('properties', {}).get('nom_commune', '')
                    if ville:
                        villes.append(ville.strip().lower().replace("-", " "))
            elif chemin_fichier.endswith('.csv'):
                import csv
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    ville = row.get('nom_commune', '')
                    if ville:
                        villes.append(ville.strip().lower().replace("-", " "))
        print(f"[INFO] {len(set(villes))} communes chargées depuis le fichier.")
    return list(set(villes))

GAZETTEER_COMMUNES_31 = charger_gazetteer("communes-haute-garonne.geojson.json")

def enlever_accents(texte):
	"""Supprime les accents d'une chaîne de caractères (ex: 'é' devient 'e')."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texte)
        if unicodedata.category(c) != 'Mn'
    )

def commune_to_regex(nom_commune):
	"""
    Transforme un nom de ville strict en une expression régulière (Regex) tolérante.
    Permet de détecter la ville même si la transcription (ASR) fait des fautes d'accents
    ou oublie des tirets (ex: "Saint-Gaudens" = "saint gaudens" = "St Gaudens").
    """
    base = enlever_accents(nom_commune.lower().strip())
    base = re.sub(r'[\s\-]+', r'[\\s-]+', base)
    mapping = {
        'e': '[eéèêë]',
        'a': '[aàâä]',
        'o': '[oôö]',
        'i': '[iîï]',
        'u': '[uùûü]',
        'c': '[cç]'
    }
    return "".join(mapping.get(c, c) for c in base)

# Création d'une énorme Regex combinant toutes les communes du 31
patterns_communes = list(set(commune_to_regex(c) for c in GAZETTEER_COMMUNES_31))
REGEX_COMMUNES = r"\b(" + "|".join(patterns_communes) + r")\b"

# ============================================================
# CHARGEMENT DE L'ENTITY RULER (Dictionnaire métier CISU)
# ============================================================
fichier_jsonl = "patterns_sdis_cisu.jsonl"
if os.path.exists(fichier_jsonl):
	# Ajoute nos règles issues du CSV dans le pipeline spaCy AVANT la détection d'entités par défaut (NER)
    ruler = nlp.add_pipe("entity_ruler", name="cisu_ruler", before="ner")
    ruler.from_disk(fichier_jsonl)
    print(f"[INFO] EntityRuler métier chargé depuis '{fichier_jsonl}'")
else:
    print(f"[AVERTISSEMENT] '{fichier_jsonl}' introuvable.")

# ============================================================
# ONTOLOGIE MÉTIER (Regex CORRIGÉES)
# ============================================================
# Ces Regex servent de "filet de sécurité" pour attraper des tournures de phrases 
# très spécifiques qui pourraient échapper à spaCy.
CONCEPTS_URGENCE = {
    "CONSCIENCE_NEURO": [
        r"(?i)\bne\s+(?:va|se\s+sent)\s+pas\s+bien\b", 
        r"(?i)\bne\s+r[eé]agit\s+pas\b", 
        r"(?i)\bne\s+bouge\s+(?:plus|pas)\b",
        r"(?i)\bne\s+(?:me\s+)?r[eé]pond(?:s|ent|ait)?\s+pas\b",
        r"(?i)\b(?:allong[eé]e?s?|couch[eé]e?s?|[eé]tendue?s?|tomb[eé]e?s?)(?:\s+(?:au\s+sol|par\s+terre))?\b",
        r"(?i)\b(?:inertes?|allit[eé]e?s?)\b",
        r"(?i)\bpertes?\s+de\s+(?:connaissance|conscience|[eé]quilibre)\b", 
        r"(?i)\b(?:transpir(?:ations?|e(?:nt|r)?)|moiteurs?|moites?|(?:gouttes?\s+de\s+)?sueurs?)\b",
        r"(?i)\btremblements?\b", 
        r"(?i)\b(?:il|elle|tu|je)\s+trembles?\b", 
        r"(?i)\bcrises?\s+(?:de\s+(?:tremblements?|spasmophilie|t[eé]tanie)|d[’'][eé]pilepsie)\b", 
        r"(?i)\bmalaises?\b", 
        r"(?i)\bparalys(?:[eé]e?s?|ies?)\b", 
        r"(?i)\byeux\s+fixes\b",
        r"(?i)\btroubles?\s+(?:sensoriels?|d[’'][eé]locution|de\s+la\s+vision)\b", 
        r"(?i)\bconvulsions?\b", 
        r"(?i)\bpupilles\s+dilat[eé]es?\b", 
        r"(?i)\bvision\s+troubl[eé]e?\b", 
        r"(?i)\bvomissements?\b", 
        r"(?i)\bvertiges?\b", 
        r"(?i)\bm(?:al|aux)\s+de\s+t[êe]te\b", 
        r"(?i)\bsous\s+(?:emprises?\s+de\s+)?(?:drogues?|alcool|stup[eé]fiants?|m[eé]dicaments?|traitements?)\b", 
        r"(?i)\b[eé]lectrocut(?:ions?|[eé]e?s?)\b", 
        r"(?i)\b(?:ingestions?|inhalations?|injections?)\s+(?:de\s+|d['’]\s*|des\s+)(?:produits?\s+(?:toxiques?|chimiques?|pharmaceutiques?|caustiques?)|m[eé]dicaments?|acides?|caustiques?|d[eé]tergents?)\b",
        r"(?i)\b[eé]vanoui(?:ssements?|e?s?)\b",  
        r"(?i)\bfrissons?\b", 
        r"(?i)\bp[aâ]leurs?\b", 
        r"(?i)\bblanch(?:eurs?|es?|s?)\b", 
        r"(?i)\blivides?\b",
        r"(?i)\b(?:anxi[eé]t[eé]s?|angoisses?|agitations?)\b",
        r"(?i)\b(?:faiblesses?|somnolences?|apathies?|l[eé]thargies?)\b",
        r"(?i)\b(?:picotements?|d[eé]mangeaisons?)\b"
    ],
    "DETRESSE_VITALE": [
        r"(?i)\b(?:flaques?\s+de\s+sang|baigne\s+dans\s+son\s+sang|se\s+vide\s+de\s+son\s+sang|saigne\s+abondamment|sang)\b",
        r"(?i)\bsaignements?(?:\s+(?:des\s+oreilles|par\s+la\s+bouche|abondants?|en\s+jet|en\s+saccades?|sous\s+pression))?\b",
        r"(?i)\b(?:coupures?|entailles?|taillader?(?:\s+les\s+veines)?|plaies?\s+profondes?|objet\s+dans\s+la\s+plaie)\b",
        r"(?i)\b(?:br[ûu]lures?|an[eé]mi[eé]e?s?|h[eé]morragies?|m[eé]trorr\s?agies?|œd[èe]mes?|oed[èe]mes?)\b",
        r"(?i)\b(?:piq[uû]res?|morsures?)(?:\s+d[e'’]\s*(?:gu[eêè]pes?|abeilles?|frelons?|insectes?))?\b",
        r"(?i)\b(?:difficult[eé]s?\s+[àa]\s+respirer|ne\s+respire\s+pas|asphyxies?|toux|s['’][eé]touff(?:e|ent|er)|[eé]touffements?|respirations?\s+rapides?|respire\s+(?:avec\s+difficult[eé]|avec\s+bruit|rapidement|bruyamment)|cr[eé]pitements?|(?:re)?cherches?\s+son\s+air|hal[èe]te(?:nt|s|ments?)?)\b",
        r"(?i)\b(?:asthmes?|insuffisances?\s+respiratoires?(?:\s+chroniques?)?|pendaisons?|[eé]tranglements?)\b",
        r"(?i)\b(?:canules?|trach[eé]otomies?)\b",
        r"(?i)\b(?:son\s+(?:c[œo]ur|coeur)\s+ne\s+bat\s+pas|malades?\s+du\s+(?:c[œo]ur|coeur)|malades?\s+cardiaques?)\b",
        r"(?i)\b(?:douleurs?\s+dans\s+la\s+poitrine|(?:la\s+)?poitrine\s+(?:qui\s+(?:se\s+)?serre|dans\s+un\s+[eé]tau))\b",
        r"(?i)\b(?:a\s+d[eé]j[àa]\s+fait\s+un\s+)?infarctus\b",
        r"(?i)\b(?:le\s+)?pouls\s+(?:est\s+)?(?:filant|rapide|(?:ir)?r[eé]gulier|(?:im)?perceptible)\b",
        r"(?i)\b(?:maladies?\s+d[e'’\s]+(?:alzheimer|lewy|parkinson|charcot|crohn|lyme)|alzheimer|parkinson|lewy|corps\s+de\s+lewy|crohn|lyme)\b",
        r"(?i)\b(?:paludismes?|tuberculoses?|diab[èe]tes?|diab[eé]tiques?|cancers?|ob[eé]sit[eé]s?|sida|vih|thromboses?|emboll?ies?|ivg|m[eé]ningites?|coqueluches?|l[eé]gionn?ell?oses?|varioles?|chikungunya|dengue|hantavirus)\b",
        r"(?i)\b(?:(?:hyper|hypo)(?:glyc[eé]mies?|tensions?)|bronch(?:iol)?ites?|(?:cardio|art[eé]rio)pathies?|gastro[- ]?ent[eé]rites?)\b",
        r"(?i)\b(?:h[eé]patites?(?:\s+(?:[a-e]|virales?|et|,))*|interruptions?\s+volontaires?\s+d[e'’\s]+grossesses?|scl[eé]roses?\s+en\s+plaques?|fi[èe]vres?\s+jaunes?)\b",
        r"(?i)\b(?:j['’]ai\s+entrepris\s+une\s+r[eé]animation|j['’]ai\s+pos[eé]\s+un\s+dae|je\s+fais\s+un\s+massage\s+cardiaque)\b",
        r"(?i)\b(?:je\s+l['’]ai\s+mis\s+(?:sur\s+le\s+c[oô]t[eé]|en\s+pls)|position\s+lat[eé]rale\s+de\s+s[eé]curit[eé])\b",
        r"(?i)\b(?:j['’]ai\s+(?:pos[eé]|mis)\s+un\s+(?:pansement(?:\s+compressif)?|garrot)|un\s+garrot|j['’]appuie\s+sur\s+(?:le\s+saignement|l['’]h[eé]morragie|la\s+plaie))\b",
        r"(?i)\b(?:retir[eé]e?s?\s+(?:du\s+danger|de\s+la\s+route|de\s+la\s+pi[èe]ce\s+enfum[eé]e?s?)|j['’]ai\s+(?:pos[eé]\s+une\s+couverture|couvert(?:e|s)?))\b"
    ],
    "TRAUMATISME": [
        r"(?i)\bchutes?(?:\s+(?:de\s+sa\s+hauteur|des\s+escaliers|d[’']\s*une\s+[eé]chelle|sur\s+la\s+t[êe]te))?\b",
        r"(?i)\bchocs?\s+sur\s+la\s+t[êe]te\b",
        r"(?i)\bd[eé]formations?(?:\s+(?:du\s+visage|de\s+la\s+bouche))?\b", 
        r"(?i)\b(?:fortes?\s+)?douleurs?(?:\s+(?:au\s+dos|vives?))?\b",
        r"(?i)\b(?:cass[eé]e?s?|d[eé]plac[eé]e?s?|amput[eé]e?s?|coup[eé]e?s?)\b",
        r"(?i)\bincapables?\s+(?:de|à)\s+bouger|incapacit[eé]s?\s+[àa]\s+bouger\b",
        r"(?i)\bamputations?\b", 
        r"(?i)\bluxations?\b", 
        r"(?i)\bd[eé]bo[îi]te(?:r|ments?)?\s+l[’']articulation\b",
        # Localisations anatomiques déplacées ici 
        r"(?i)\b(?:cr[âa]nes?|t[êe]tes?|cous?|(?:[œo]ils?|yeux)|oreilles?|nez|bouches?|l[èe]vres?)\b",
        r"(?i)\b(?:dos|c[ôo]tes?|thorax|poitrines?|ventres?|abdomens?)\b",
        r"(?i)\b(?:[eé]paules?|bras|mains?|doigts?|aines?|cuisses?|jambes?|pieds?)\b"
    ],
    "INCENDIE_GAZ": [
        r"(?i)\b(?:panaches?\s+de\s+)?fum[eé]es?(?:\s+(?:opaques?|noires?|grises?|blanches?|irrespirables?|qui\s+piquent?\s+les\s+yeux|s['’][eé]chappent?))?\b",
        r"(?i)\b(?:incendies?|feux?|boules?\s+de\s+feu|foyers?|brasiers?|torch[eè]res?|lueurs?|[eé]tincelles?)\b",
        r"(?i)\b(?:feux?\s+isol[eé]s?)\b",
        r"(?i)\b(?:flammes?|en\s+flammes?|cr[eé]pitements?(?:\s+des\s+flammes)?|chaleurs?\s+intenses?)\b",
        r"(?i)\b(?:compteurs?\s+(?:[eé]lectriques?|d[e'’]\s*gaz)|batteries?(?:\s+[eé]lectriques?)?)\b",
        r"(?i)\b(?:poubelles?|mobiliers?\s+urbains?|(?:(?:feux?\s+d[e'’]\s*(?:un\s+|des\s+)?)?(?:meubles?|matelas)))\b",
        r"(?i)\bodeurs?\s+(?:de\s+(?:gaz|br[ûu]l[eé])|naus[eé]abondes?)\b",
        r"(?i)\bgaz\s+(?:toxiques?|irritants?)\b",
        r"(?i)\bfuites?\s+de\s+gaz(?:\s+enflamm[eé]e?s?)?\b",
        r"(?i)\b(?:(?:en\s+train\s+de\s+)?br[ûu]le(?:r|nt|s)?)\b",
        r"(?i)\b(?:[cç][aà]\s+flambe|allumer|enflammer|incendier|embraser)\b",
        r"(?i)\b(?:s['’]est\s+)?embras[eé]e?s?\b",
        r"(?i)\b(?:explosions?|intoxications?|fissures?|effondrements?)\b"
    ],
    "ACCIDENT_ROUTE": [
        r"(?i)\baccidents?(?:\s+(?:routiers?|graves?|simples?|l[eé]gers?))?\b",
        r"(?i)\bcarambolages?\b",  
        r"(?i)\btonneaux?\b",
        r"(?i)\bchocs?\s+(?:front(?:al|aux)|lat[eé]r(?:al|aux)|(?:par\s+)?l['’]arri[eè]re)\b",
        r"(?i)\bvoitures?\s+(?:compl[eè]tement\s+)?(?:d[eé]form[eé]e?s?|explos[eé]e?s?|d[eé]truite?s?|[eé]cras[eé]e?s?)\b",
        r"(?i)\b(?:personnes?\s+)?(?:[eé]ject[eé]e?s?|coinc[eé]e?s?|sorties?|incarc[eé]r[eé]e?s?|pi[eé]g[eé]e?s?)\b",
        r"(?i)\b(?:(?:absence\s+de|port\s+d[eu])\s+casques?|casques?\s+arrach[eé]s?)\b",
        r"(?i)\b(?:(?:port|prot(?:ections?)?\.?)\s+des\s+gants|absence\s+de\s+protections?\s+corporelles?)\b",
        r"(?i)\b(?:grandes?|petites?)\s+vitesses?\b", 
        r"(?i)\bcin[eé]tiques?\s+importantes?\b",
        r"(?i)\b(?:voitures?|bus|tracteurs?|scooters?|v[eé]los?|quads?|motos?)\b",
        r"(?i)\bpoids\s+lourds?\b", 
        r"(?i)\btrottinettes?(?:\s+[eé]lectriques?)?\b", 
        r"(?i)\bcamions?(?:\s+(?:petits?|gros?))?\b",
        r"(?i)\b(?:diesel|essences?|super\s+carburants?|gaz|gpl|carburants?\s+verts?|[eé]thanol)\b",
        r"(?i)\b(?:(?:carburations?\s+)?hybrides?|[eé]lectriques?)\b", 
        r"(?i)\b(?:suvs?|4\s*[xX*]\s*4|quatre\s*[- ]?\s*quatre)\b"
    ],
    "LOCALISATION_SPECIFIQUE": [
        r"(?i)\b(?:en\s+ville|(?:aux?|un|des|les?)\s+(?:ronds?[- ]points?|croisements?))\b",
        r"(?i)\b(?:autoroutes?|voies?(?:\s+(?:rapides?|publiques?|ferr[eé]es?|fluviales?))?|pistes?\s+cyclables?)\b",
        r"(?i)\b(?:ponts?|p[eé]ages?|tunnels?)\b",
        r"(?i)\b(?:gares?(?:\s+(?:ferroviaires?|ferrovi[èe]res?|routi[èe]res?))?|ports?|quais?|embarcad[èe]res?|a[eé]ro(?:ports?|dromes?))\b",
        r"(?i)\bdans\s+(?:un|des|les?)\s+virages?\b",
        r"(?i)\bdans\s+(?:une|des|les?)\s+courbes?\b",
        r"(?i)\ben\s+lignes?\s+droites?\b",
        r"(?i)\bterres?[- ]pleins?(?:\s+centra(?:l|ux))?\b",
        r"(?i)\b(?:barri[èe]res?|glissi[èe]res?|rambardes?)(?:\s+de\s+s[eé]curit[eé])?\b",
        r"(?i)\b(?:sur\s+le\s+)?trottoirs?\b",
        r"(?i)\bdans\s+(?:un|le|les?|des)\s+(?:foss[eé]s?|pr[eé]cipices?)\b",
        r"(?i)\b(?:en\s+contre[- ]bas|abrupte?s?|d[eé]nivel[eé]s?\s+important(?:e?s?)?)\b",
        r"(?i)\b[aà]\s+quitt[eé](?:r|s?|es?)?\s+la\s+route\b",
        r"(?i)\bcontre\s+(?:un|des|les?)\s+arbres?\b",
        r"(?i)\b(?:contre\s+)?(?:un|des|les?)\s+rochers?\b",
        r"(?i)\b(?:contre\s+)?(?:un|des|les?)\s+(?:murs?|murets?|poteaux?|lampadaires?|pyl[ôo]nes?)\b",
        r"(?i)\b(?:domiciles?|b[aâ]timents?|hlm|pavillons?|maisons?(?:\s+particuli[èe]res?)?(?:\s+(?:de\s+ville|en\s+campagne))?|habitations?\s+collectives?|foyers?\s+d['’]\s*h[eé]bergements?)\b",
        r"(?i)\b(?:portails?|digicodes?|r[eé]sidences?\s+s[eé]curis[eé]e?s?|(?:chemins?|acc[èe]s)\s+(?:difficiles?|impraticables?)|impraticables?)\b",
        r"(?i)\b(?:rez[- ]de[- ]chauss[eé]e|rdc|niveau\s+(?:z[eé]ro|0))\b",
        r"(?i)\b[àa]\s+l['’][eé]tage\b",
        r"(?i)\b(?:\d{1,2}(?:er|ème|eme|e)?|premier|deuxi[èe]me|troisi[èe]me|quatri[èe]me|cinqui[èe]me|sixi[èe]me|septi[èe]me|huiti[èe]me|neuvi[èe]me|dixi[èe]me|onzi[èe]me|douzi[èe]me|un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze)\s+[eé]tages?\b",
        r"(?i)\b[eé]tages?\s+(?:\d{1,2}|un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze)\b",
        r"(?i)\bappart(?:ement)?\s+(?:num[eé]ro\s+|n[°o]\s+)?(\d{1,4}(?:\s*[a-z]+)?|[a-z])\b",
        r"(?i)\b(?:autoroute|d[eé]partementale|nationale|rn|rd)\s*[-]?\s*(?:\d+|(?:(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|et)\s*[-]?\s*)+)\b",
        r"\b(?:(?i:[dnl])\s*\d{1,4}|[aA]\s*\d{1,4})\b",
        r"(?i)\b(?:d|n)\s+(?:(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|et)\s*[-]?\s*)+\b",
        r"(?i)\b(?:(?:(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|et)(?:[\s-]+(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|et))*|\d{1,4})(?:\s*(?:bis|ter|quater|[a-z]))?\s*,?\s+)?(?:r[eé]sidence|chemin|rue|avenue|impasse|place|b[aâ]timent|boulevard|all[eé]e|route|voie|cours|quai|lieux?[- ]dits?|quartiers?)(?:(?:\s+|-)(?:(?:de\s+la|de\s+l['’]|des|du|de|la|le|les|l['’]|d['’])\s+)?(?!(?:et|il|elle|je|tu|on|nous|vous|ils|elles|dans|devant|derri[èe]re|autour|avec|pour|qui|que|quoi|est|sont|ont|[cç]a?|ce|cette|ces|ceux|celles?|tout|toute?s?|tr[èe]s|mais|donc|car|puis|ensuite|voil[àa]|l[àa]|par|ici|vite|y|en|un|une)\b)[a-zà-âçéèêëîïôûùü0-9]+){0,4}\b",
        r"(?i)\b(?:petites?\s+rues?|rues?\s+[eé]troites?|chemins?\s+de\s+terre|impasses?|tunnels?|ruelles?|passages?|b[aâ]timents?|maisons?)\b",
        r"(?i)\b(?:torrents?|rivi[èe]res?|lacs?|fleuves?|trous?|puits|foss[eé]s?\s+profonds?|ruisseaux?|cours?\s+d['’]eau)\b"
    ],
    "COMMUNES_31": [
        REGEX_COMMUNES 
    ],
    "INTERVENTIONS_Diverses": [
        r"(?i)\b(?:chats?|chiens?|b[eé]tails?|gu[êèée]pes?|abeilles?|frelons?)\b",
        r"(?i)\b(?:anim(?:al|aux)|chats?|chiens?)\s+(?:bl[eé]ss[eé]s?|morts?|[eé]cras[eé]s?)\b",
        r"(?i)\b(?:animal|animaux)\s+(?:sauvages?|domestiques?|dangereux|d['’]\s*[eé]levages?)\b", 
        r"(?i)\b(?:(?:arbres?|branches?)\s+(?:arrach[eé]e?s?|cass[eé]e?s?|tomb[eé]e?s?)|(?:toitures?|tuiles?)\s+(?:arrach[eé]e?s?|(?:qui\s+)?s['’]envolent)|vitres?\s+cass[eé]e?s?)\b",
        r"(?i)\b(?:mati[èe]res?|produits?|gaz|vapeurs?|solides?|liquides?)\s+(?:radiologiques?|radioacti(?:f|ves?)|chimiques?|explosi(?:f|ves?)|irritantes?|irritants?|corrosi(?:f|ves?)|toxiques?)\b",
        r"(?i)\b(?:fuites?|d[eé]versements?|[eé]coulements?)(?:\s+d[e'’]\s*(?:gaz|liquides?|mati[èe]res?|produits?|chimiques?|toxiques?))?\b",
        r"(?i)\b(?:ouvertures?\s+de\s+portes?|pertes?\s+de\s+cl[eé]s?|ascenseurs?\s+bloqu[eé]s?)\b",
        r"(?i)\b(?:dangers?|inondations?|d[eé]bordements?)\b",
        r"(?i)\b(?:enceintes?|[àa]\s+terme|(?:menaces?\s+d['’]\s*)?accouchements?|d[eé]buts?\s+d[eu]\s+travail|pertes?\s+des\s+eaux)\b", 
        r"(?i)\b(?:douleurs?\s+(?:abdominales?|(?:au\s+)?bas[- ]?ventres?|rapproch[eé]e?s?(?:\s+(?:et\s+)?r[eé]guli[èe]res?)?))\b",
        r"(?i)\b(?:probl[èe]mes?\s+gyn[eé]cologiques?)\b"
    ],
    "METEO":[
        r"(?i)\b(?:froids?|froid[es]?|chauds?|chaud[es]?|s[eé]cheresses?)\b",
        r"(?i)\b(?:brouillards?|neiges?|verglas|glaces?)\b",
        r"(?i)\b(?:temp[êeè]tes?|tornades?|orages?\s+violents?|vents?\s+(?:forts?|violents?|d['’]\s*autan)|mistral|tramontanes?|gr[êeè]l(?:es?|ons?))\b"
    ],
    "ARMES_VIOLENCE": [
        r"(?i)\barmes?\s+[aà]\s+feu\b",
        r"(?i)\bfusils?\b",
        r"(?i)\bpistolets?\b",
        r"(?i)\bkalas[ch]nikovs?\b",
        r"(?i)\bcouteaux?\b",
        r"(?i)\bhaches?\b",
        r"(?i)\bgourdins?\b",
        r"(?i)\bbattes?\s+de\s+base[-\s]?ball\b"
    ],
    "IMPLIQUES": [
        # Fusion et standardisation de tous les individus
        r"(?i)\b(?:enfants?|b[eé]b[eé]s?|adultes?|p[èe]res?|m[èe]res?|fils|filles?|gar[çc]ons?|adolescent(?:e?s?)?|voisin(?:e?s?)?|personnes?|patient(?:e?s?)?|nourrissons?|grand[- ]p[èe]res?|grand[- ]m[èe]res?|malades?)\b",
        r"(?i)\bpersonnes?\s+(?:[âa]g[eé]e?s?|handicap[eé]e?s?)\b", 
        r"(?i)\b(?:avec\s+un\s+)?handicaps?\b", 
        r"(?i)\b(?:en\s+)?fauteuils?\s+roulants?\b",
        r"(?i)\b(?:pi[eé]tons?|marcheurs?|randonneurs?|cyclistes?|motards?)\b",
        r"(?i)\b(?:maires?|conseill(?:er|ère)s?|d[eé]put[eé]e?s?|s[eé]nat(?:eur|rice)s?|ministres?)\b",
        r"(?i)\b(?:(?:pr[eé]sences?\s+d['’]\s*)?autorit[eé]s?|consuls?|ambassad(?:eur|rice)s?)\b",
        r"(?i)\b(?:polici(?:er|ère)s?|gendarmes?|(?:sapeurs?[- ]?)?pompiers?|soldats?|militaires?|agents?\s+publics?)\b"
    ],
    "AGE_VICTIME": [
        r"(?i)\b(\d+|(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|et)(?:[\s-]+(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|et))*)\s+(ans?|mois|jours?|semaines?)\b"
    ],
    "MESURES": [
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:m[eè]tres?|centim[eè]tres?|millim[eè]tres?|kilom[eè]tres?|km|cm|mm|m)\b",
        r"(?i)\b(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|dizaine|centaine|et|de)(?:[\s-]+(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|dizaine|centaine|et|de))*\s+(?:m[eè]tres?|centim[eè]tres?|millim[eè]tres?|kilom[eè]tres?|km|cm|mm|m)\b",
        
        r"(?:\b0|\+33\s?)[1-9](?:[\s.-]?\d{2}){4}\b",
        r"\b\d{5}\b"
    ],
    "DEMANDE_MOYENS": [
        r"(?i)\bdemandes?\s+(?:d[’']ambulances?|de\s+secours)\b", 
        r"(?i)\b(?:m[eé]decins?|docteurs?|infirmi[eèé]re?s?)\b", 
        r"(?i)\b(?:camions?|voitures?|v[eé]hicules?)\s+(?:de\s+feu|pour\s+combattre\s+l[’']incendie|de\s+pompiers?|de\s+secours)\b", 
        r"(?i)\blutter\s+contre\s+l[’']incendie\b",
        r"(?i)\b(?:gros|petits?)\s+camions?\b", 
        r"(?i)\b(?:(?:grandes?|petites?)\s+[eé]chelles?|[eé]chelles?\s+(?:manuelles?|[aà]\s+mains?))\b"
    ]
}
# ============================================================
# CONTEXTES ONTOLOGIE
# ============================================================
# Définit les mots qui renforcent (mots_clefs) ou annulent (mots_penalisants)
# la probabilité qu'un événement métier soit réel, en analysant la phrase autour du mot.
CONTEXTES_ONTOLOGIE = {
    "INCENDIE_URBAIN": {
        "mots_clefs": ["maison", "appartement", "immeuble", "bâtiment", "toit", "fenêtre", "porte", "garage", "cuisine", "chambre",
        "salon", "tableau électrique", "compteur électrique", "cheminée", "hlm", "copropriété"],
        "mots_penalisants": ["barbecue", "encens", "bougie", "vapeur", "cigarette", "mégot", "cuisine", "four", "poêle", "radiateur", "chaudière", "bain", "douche"],
        "poids": 2.0,
        "poids_negatif": -2.0
    },
    "INCENDIE_VEHICULE": {
        "mots_clefs": ["voiture", "camion", "moto", "scooter", "bus", "carburant", "essence", "gas oil", "poids lourd", "moteur", "capot",
        "réservoir", "batterie", "parking", "véhicule", "4x4", "suv"],
        "mots_penalisants": ["pot d'échappement", "vapeur", "démarrage", "radiateur", "condensation", "froid", "hiver"],
        "poids": 2.0,
        "poids_negatif": -1.5
    },
    "INCENDIE_FORET": {
        "mots_clefs": ["forêt", "bois", "arbre", "maquis", "garrigue", "sécheresse", "végétation", "broussailles", "pinède"],
        # On pénalise l'incendie de forêt si la fumée sort d'un équipement domestique
        "mots_penalisants": ["cheminée", "barbecue", "brouillard", "brume", "poussière", "tracteur", "moissonneuse", "labour", "nuage"], 
        "poids": 1.8,
        "poids_negatif": -2.0 # ex : Détruit le score "Forêt" si on parle d'une cheminée
    },
    "INCENDIE_INDUSTRIEL": {
        "mots_clefs": ["usine", "palettes", "hydrocarbures", "zone industrielle", "entrepôt", "atelier", "chantier", "site", "stockage", "dépôt", "hangar", "siloz"],
        "mots_penalisants" : ["torchère", "vapeur", "cheminée d'usine", "refroidissement", "test", "essai", "maintenance", "contrôle", "exercice interne"],
        "poids": 1.8,
        "poids_negatif": -1.5
    },
    "BRULAGE_CONTROLE": {
        "mots_clefs": ["brûlage", "écobuage", "déchets", "feuilles mortes", "barbecue", "végétaux", "jardin", "feu de camp", "cigarette", "mégot"],
        "poids": 1.5
    },
    "ACCIDENT_ROUTIER": {
        "mots_clefs": ["voiture", "camion", "moto", "scooter", "bus", "poids lourd", "véhicule", "accident", "choc", "collision"],
        "mots_penalisants" : ["matériel", "constat", "amiable", "carrosserie", "rayure", "rétroviseur", "stationnement", "parking", "auto-tamponneuse", "jeu vidéo", "circuit"],
        "poids": 1.5,
        "poids_negatif": -2.0
    },
    "DETRESSE_MEDICALE": {
        "mots_clefs": ["malade", "blessé", "inconscient", "chute", "malaise", "cardiaque", "infarctus"],
        "mots_penalisants": ["dort", "sommeil", "ivre", "bourré", "alcoolisé", "cuite", "formation", "secourisme", "manœuvre", "sst", "psc1", "rendez-vous", "consultation"],
        "poids": 1.5,
        "poids_negatif": -2.0
    },
    "NUISANCE_ODORANTE": {
        "mots_clefs": [
            "odeur", "gaz", "émanation", "fumée", "brûlé", "cramé", "chimique",
            "grise", "grises", "blanche", "blanches", "noire", "noires",
            "épaisse", "épaisses", "opaque", "opaques", "toxique", "toxiques",
            "qui pique", "irrespirable", "irrespirables"],
        "mots_penalisants": ["lisier", "fumier", "épandage", "agriculture", "égout", "poubelle", "déchetterie", "parfum", "cuisine"],    
        "poids": 1.2,
        "poids_negatif": -1.5
    },
    
    # ---  FACTEURS AGGRAVANTS ---
    
    "FACTEUR_AGGRAVANT_ACCIDENT": {
        "mots_clefs": ["en contre bas", "tonneaux", "tonneau", "sortie de route", "choc frontal", "choc violent", "éjecté", "verglas", "brouillard"],
        "poids": 2.0
    },
    "FACTEUR_AGGRAVANT_INCENDIE": {
        "mots_clefs": ["bouteille de gaz", "gaz", "explosion", "matières dangereuses", "matière dangereuse"],
        "poids": 2.0
    },
    "FACTEUR_AGGRAVANT_MALAISE": {
        "mots_clefs": ["nourrisson", "personne âgée", "antécédent médical", "femme enceinte", "malade cardiaque", "asthme", "ne respire pas", "respire difficilement"],
        "poids": 2.0
        
    }
}

# ============================================================
# FONCTIONS DE NORMALISATION
# ============================================================
def normalize_text_for_detection(text):
	"""
    Prépare le texte pour la détection IA.
    Élimine les "bruits" oraux (disfluences) fréquents dans les appels téléphoniques
    pour éviter qu'ils ne perturbent l'analyse sémantique.
    """
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Liste des mots d'hésitation typiques de l'oral (ASR noise)
    disfluences = {
        "ah", "aie", "atchoum", "baf", "bah", "be", "ben", "bien", "bof",
        "bon ben", "bouh", "euh", "euf", "ha", "heu", "heueu", "he", "he bien",
        "hein", "hi", "ih", "hm", "hop", "hou", "hum", "hup", "la", "mah", 
        "menfin", "mmm", "mouais", "moui", "of", "oh", "ok", "okay", "ouah", 
        "ouais", "ouf", "ouille", "pff", "pouh", "snif", "tac", "toc", 
        "wahou", "yeah", "zut", "zou", "und"
    }
    
    # Trie par longueur décroissante pour éviter qu'un bout de mot ne soit effacé
    for df in sorted(disfluences, key=len, reverse=True):
        pattern = r"\b" + re.escape(df) + r"\b"
        text = re.sub(pattern, " ", text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text_for_display(text):
	"""Nettoyage léger pour l'affichage humain dans la console."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# EXTRACTION DES MOTS MÉTIER AVEC SPACY
# ============================================================
def extract_metier_words_with_spacy(text):
	"""
    Fonction centrale d'extraction. Fonctionne en 3 passes successives :
    Passe 1 : Expressions Régulières (Regex) - pour les expressions figées.
    Passe 2 : EntityRuler - pour le dictionnaire métier issu des CSV (JSONL).
    Passe 3 : spaCy NER - Modèle IA probabiliste pour détecter des noms propres ou lieux inconnus.
    """
    matches = []
    text_lower = text.lower()
    
    # PASSE 1 : Regex métier
    for categorie, patterns in CONCEPTS_URGENCE.items():
        for pattern in patterns:
            try:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    mot_exact = match.group(0).strip()
                    matches.append({
                        'categorie': categorie,
                        'mot': mot_exact,
                        'start': match.start(),
                        'end': match.end(),
                        'type': 'REGEX_SDIS',
                        'source': 'Ontologie SDIS',
                        'label_original': categorie
                    })
            except re.error as e:
                print(f"[AVERTISSEMENT] Regex invalide pour {categorie}: {pattern} - {e}")
                continue
    
    # PASSE 2 & 3 : EntityRuler + spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
		# Si le label commence par CISU_, c'est notre dictionnaire JSONL (Passe 2)
        if ent.label_.startswith("CISU_"):
            categorie_ner = ent.label_
            type_detection = 'ENTITY_RULER'
            source_detection = "Dictionnaire CISU Excel"
            # Sinon, c'est l'IA par défaut de spaCy (Passe 3 : NER)
        elif ent.label_ in ["LOC", "ORG"]:
            categorie_ner = f"LIEU_AUTO_{ent.label_}"
            type_detection = 'SPACY_NER'
            source_detection = f"spaCy NER ({ent.label_})"
        elif ent.label_ in ["PER", "PERSON"]:
            categorie_ner = f"PERSONNE_AUTO_{ent.label_}"
            type_detection = 'SPACY_NER'
            source_detection = f"spaCy NER ({ent.label_})"
        elif ent.label_ in ["DATE", "TIME"]:
            categorie_ner = f"TEMPORALITE_AUTO_{ent.label_}"
            type_detection = 'SPACY_NER'
            source_detection = f"spaCy NER ({ent.label_})"
        elif ent.label_ == "CARDINAL":
            categorie_ner = f"QUANTITE_AUTO_{ent.label_}"
            type_detection = 'SPACY_NER'
            source_detection = f"spaCy NER ({ent.label_})"
        elif ent.label_ == "MISC":
            categorie_ner = f"DIVERS_AUTO_{ent.label_}"
            type_detection = 'SPACY_NER'
            source_detection = f"spaCy NER ({ent.label_})"
        else:
            categorie_ner = f"AUTRE_AUTO_{ent.label_}"
            type_detection = 'SPACY_NER'
            source_detection = f"spaCy NER ({ent.label_})"
        
        texte_entite = ent.text.strip()
        
        if len(texte_entite) < 3:
            continue # Ignore les mots trop courts
        
        # SÉCURITÉ : Évite les doublons si la Regex et spaCy ont trouvé le même mot
        est_deja_present = False
        for existing in matches:
            if existing['mot'] == texte_entite.lower():
                est_deja_present = True
                break
        
        if not est_deja_present:
            matches.append({
                'categorie': categorie_ner,
                'mot': texte_entite,
                'start': ent.start_char,
                'end': ent.end_char,
                'type': type_detection,
                'source': source_detection,
                'label_original': ent.label_
            })
    
    # Dédoublonnage final basé sur le tuple (Catégorie, Mot)
    matches_unis = []
    vus = set()
    for m in matches:
        cle = (m['categorie'], m['mot'].lower())
        if cle not in vus:
            vus.add(cle)
            matches_unis.append(m)
    
    return matches_unis

# ============================================================
# CHARGEMENT DES FICHIERS
# ============================================================
def load_whisperx_full_text(filepath):
	"""Charge le fichier de sortie de l'IA de transcription (WhisperX)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tous_les_mots = []
    for segment in data.get('segments', []):
        if 'words' in segment:
            for word_info in segment['words']:
                tous_les_mots.append({
                    'word': word_info['word'],
                    'start': word_info.get('start', 0),
                    'end': word_info.get('end', 0)
                })
    
    tous_les_mots.sort(key=lambda x: x['start'])
    texte_complet = " ".join([w['word'] for w in tous_les_mots])
    
    return {
        'full_text': texte_complet,
        'words': tous_les_mots
    }

def load_elan_full_text(filepath):
	"""Charge le fichier de référence créé par un humain (Ground Truth ELAN)."""
    eaf = pympi.Elan.Eaf(filepath)
    tous_tiers = eaf.get_tier_names()
    
    MOTS_A_IGNORER = {"bruit", "und", "[bruit]", "<bruit>", "(bruit)"}
    
    toutes_annotations = []
    for tier in tous_tiers:
        if tier in eaf.get_tier_names():
            for debut_ms, fin_ms, texte in eaf.get_annotation_data_for_tier(tier):
                if texte and texte.strip():
                    texte_propre = texte.strip()
                    
                    if texte_propre.lower() in MOTS_A_IGNORER:
                        continue
                        
                    # Nettoyage des balises sonores parasites
                    texte_propre = re.sub(r'\bund\b', '', texte_propre, flags=re.IGNORECASE)
                    texte_propre = re.sub(r'\[bruit\]|\(bruit\)|<bruit>', '', texte_propre, flags=re.IGNORECASE)
                    texte_propre = re.sub(r'\s+', ' ', texte_propre).strip()
                    
                    if texte_propre:
                        toutes_annotations.append({
                            'start': debut_ms / 1000.0,
                            'end': fin_ms / 1000.0,
                            'text': texte_propre
                        })
    
    toutes_annotations.sort(key=lambda x: x['start'])
    texte_complet = " ".join([ann['text'] for ann in toutes_annotations])
    
    return {
        'full_text': texte_complet,
        'segments': toutes_annotations
    }

# ============================================================
# ANALYSE DES ERREURS WHISPER
# ============================================================
def analyze_whisper_error_with_spacy(metier_item, whisper_text):
	"""
    Détermine si un mot de la référence a été correctement transcrit par Whisper.
    Utilise SequenceMatcher pour tolérer de petites fautes d'orthographe de l'ASR
    (ex: "hémorragie" transcrit "émoragie" -> Classé en SUBSTITUTION avec score > 0.7).
    """
    metier_word = metier_item['mot']
    metier_norm = normalize_text_for_detection(metier_word)
    whisper_norm = normalize_text_for_detection(whisper_text)
    
    if metier_norm in whisper_norm:
        pos = whisper_text.lower().find(metier_word.lower())
        if pos != -1:
            forme_percue = whisper_text[pos:pos+len(metier_word)]
        else:
            forme_percue = metier_word
        return 'CORRECT', forme_percue, 1.0, None
    
    doc_whisper = nlp(whisper_text[:5000])
    
    for ent in doc_whisper.ents:
        similarite = SequenceMatcher(None, metier_word.lower(), ent.text.lower()).ratio()
        if similarite > 0.7:
            return 'SUBSTITUTION', ent.text, similarite, f"spaCy a détecté {ent.label_} au lieu de {metier_item.get('label_original', 'inconnu')}"
    
    mots_whisper = whisper_norm.split()
    meilleur_match = None
    meilleur_score = 0
    
    for mot in mots_whisper:
        ratio = SequenceMatcher(None, metier_norm, mot).ratio()
        if ratio > meilleur_score and ratio > 0.6:
            meilleur_score = ratio
            meilleur_match = mot
    
    if meilleur_match and meilleur_score >= 0.7:
        return 'SUBSTITUTION', meilleur_match, meilleur_score, None
        
        
    # 3. Le mot a été totalement raté par Whisper
    return 'DELETION', None, 0, None

# ============================================================
# STOPWORDS GLOBAUX
# ============================================================
# Liste globale de "Stop words" (mots vides) pour ne pas polluer l'analyse
STOPWORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 'donc', 'car',
    'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'dans',
    'par', 'sur', 'pour', 'avec', 'sans', 'chez', 'est', 'sont', 'avait', 'avais',
    'il', 'ne', 'que', 'qui', 'être', 'avoir', 'à', 'au', 'aux', 'en', 'se', 'me', 'te',
    'nous', 'vous', 'y', 'en', 'lui', 'leur', 'moi', 'toi', 'soi', 'là', 'ici', 'bien',
    'très', 'trop', 'plus', 'moins', 'aussi', 'donc', 'or', 'ni', 'car', 'mais'
}

# ============================================================
# FONCTION analyze_insertions_with_spacy (MODIFIÉE)
# ============================================================
def analyze_insertions_with_spacy(ref_text, whisper_text, metier_words):
	"""Identifie les mots 'inventés' par Whisper qui n'étaient pas dits (Hallucinations)."""
    ref_norm = set(normalize_text_for_detection(ref_text).split())
    whisper_norm = set(normalize_text_for_detection(whisper_text).split())
    
    metier_mots_ref = set()
    for m in metier_words:
        for mot in normalize_text_for_detection(m['mot']).split():
            metier_mots_ref.add(mot)
    
    insertions = whisper_norm - ref_norm
    
    # Utilisation de STOPWORDS global
    insertions_significatives = []
    for mot in insertions:
        if len(mot) > 3 and mot not in STOPWORDS and mot not in metier_mots_ref:
            doc = nlp(mot)
            if doc.ents:
                insertions_significatives.append({
                    'mot': mot,
                    'type_spacy': doc.ents[0].label_
                })
            elif len(mot) > 4:
                insertions_significatives.append({
                    'mot': mot,
                    'type_spacy': 'mot_inconnu'
                })
    
    return insertions_significatives

# ============================================================
# FONCTION analyser_contexte_autour_d_entite (MODIFIÉE)
# ============================================================
def analyser_contexte_autour_d_entite(texte, entite_start, entite_end, fenetre_mots=8):
	
    """
    Extrait une "fenêtre" de N mots avant et après une entité détectée.
    Cette fonction est cruciale : elle utilise les lemmes (racines) via spaCy 
    pour vérifier si le contexte valide l'alerte ou la pénalise.
    Analyse le contexte autour d'une entité dans le texte.
    Utilise STOPWORDS global pour filtrer les mots vides.
    fenetre_mots=8 limitation a 8 mots 
    """
    if entite_start < 0:
        entite_start = 0
    if entite_end > len(texte):
        entite_end = len(texte)
    
    # Création de la fenêtre de contexte (ex: 8 mots avant, le mot, 8 mots après)
    texte_avant = texte[:entite_start].split()
    texte_apres = texte[entite_end:].split()
    
    contexte_mots = texte_avant[-fenetre_mots:] + [texte[entite_start:entite_end]] + texte_apres[:fenetre_mots]
    contexte_texte = " ".join(contexte_mots)
    
    doc = nlp(contexte_texte)
    
    contextes_detectes = {}
    
    for nom_contexte, config in CONTEXTES_ONTOLOGIE.items():
        score = 0
        mots_trouves = []
        mots_significatifs_trouves = 0
        
        for token in doc:
            # Ignorer les stopwords (utilise STOPWORDS global)
            if token.text.lower() in STOPWORDS:
                continue
            
            token_lemma = token.lemma_.lower()# Lemmatisation (ex: "brûlés" -> "brûler")
            mot_trouve = False
            
            # 1. Vérification des mots POSITIFS
            for mot_clef in config["mots_clefs"]:
                if mot_clef in token_lemma or token_lemma in mot_clef:
                    score += 1 * config["poids"]
                    mots_trouves.append(token.text)
                    mots_significatifs_trouves += 1
                    mot_trouve = True
                    break
                    
            # 2. Vérification des mots NÉGATIFS (Pénalisation) 
            if config.get("mots_penalisants"):
                for mot_negatif in config["mots_penalisants"]:
                    if mot_negatif in token_lemma or token_lemma in mot_negatif:
                        score += 1 * config.get("poids_negatif", -1.0)
                        mots_trouves.append(f"[-]{token.text}") 
                        mots_significatifs_trouves += 1
                        mot_trouve = True
                        break
            
            if mot_trouve:
                continue
        
        # On valide le contexte uniquement si au moins 2 indices concordants sont trouvés
        mots_contexte = [t for t in doc if t.text.lower() not in STOPWORDS]
        for i in range(len(mots_contexte) - 1):
            bigramme = f"{mots_contexte[i].text} {mots_contexte[i+1].text}".lower()
            for mot_clef in config["mots_clefs"]:
                if mot_clef in bigramme:
                    score += 0.5 * config["poids"]
                    mots_trouves.append(bigramme)
                    mots_significatifs_trouves += 1
        
        # On valide le contexte uniquement si au moins 2 indices concordants sont trouvés
        if score > 0 and mots_significatifs_trouves >= 2:
            contextes_detectes[nom_contexte] = {
                "score": score,
                "mots_contextuels": list(set(mots_trouves)),
                "nb_mots_significatifs": mots_significatifs_trouves
            }
    
    return contextes_detectes

def evaluer_contexte_metier(metier_item, texte_complet, seuil_confidence=2.0):
    """Calcule un score de confiance global et lève un drapeau (flag) si Faux Positif suspecté."""
    mot = metier_item['mot']
    start = metier_item.get('start', 0)
    end = metier_item.get('end', len(mot))
    
    # Si on n'a pas les positions, on cherche le mot dans le texte
    if start == 0 and end == len(mot):
        pos = texte_complet.lower().find(mot.lower())
        if pos != -1:
            start = pos
            end = pos + len(mot)
    
    contextes = analyser_contexte_autour_d_entite(texte_complet, start, end)
    
    # Plancher à zéro : un contexte ne peut pas avoir un score final négatif, au pire il est neutre (0)
    score_total = sum(c["score"] for c in contextes.values())
    score_total = max(0, score_total)
    score_normalise = min(score_total / 5.0, 1.0)
    
    # RÈGLES MÉTIER : Détection explicite des fausses alertes (Faux positifs)
    faux_positif_probable = False
    raison_faux_positif = []
    
    # Règle : "fumée" en contexte forestier sans signe d'incendie
    if "fumée" in mot.lower() or "fume" in mot.lower():
        if "INCENDIE_FORET" in contextes and "BRULAGE_CONTROLE" not in contextes:
            if "INCENDIE_URBAIN" not in contextes and "INCENDIE_VEHICULE" not in contextes:
                faux_positif_probable = True
                raison_faux_positif.append("Fumée en contexte forestier sans signe d'incendie")
    
    # Règle : "brûle" sans contexte de feu actif
    if "brûle" in mot.lower() or "brule" in mot.lower():
        if "INCENDIE_URBAIN" not in contextes and "INCENDIE_VEHICULE" not in contextes:
            if "INCENDIE_FORET" not in contextes:
                faux_positif_probable = True
                raison_faux_positif.append("Verbe 'brûler' sans contexte de feu actif")
    
    # Interprétation
    if contextes:
        contexte_principal = max(contextes.items(), key=lambda x: x[1]["score"])
        interpretation = f"Contextualisé avec {contexte_principal[0]} (score: {contexte_principal[1]['score']:.1f})"
    else:
        interpretation = "Non contextualisé - mot isolé"
    
    return {
        "mot": mot,
        "contextes_detectes": contextes,
        "score_confidence": score_normalise,
        "interpretation": interpretation,
        "faux_positif_probable": faux_positif_probable,
        "raison_faux_positif": raison_faux_positif
    }

def analyser_tous_les_contextes(metier_words, texte_complet):
	"""Boucle d'orchestration pour l'analyse contextuelle de toutes les entités."""
    resultats_contextes = []
    
    for metier in metier_words:
        analyse = evaluer_contexte_metier(metier, texte_complet)
        analyse.update({
            "categorie": metier['categorie'],
            "type_detection": metier['type'],
            "source": metier['source']
        })
        resultats_contextes.append(analyse)
    
    return resultats_contextes

# ============================================================
# ÉVALUATION COMPLÈTE AVEC CONTEXTE
# ============================================================
def evaluate_metier_complete_avec_contexte(ref_data, whisper_data):
	"""Fonction maîtresse : Extrait, compare et annote les résultats finaux."""
    ref_text = ref_data['full_text']
    whisper_text = whisper_data['full_text']
    
    print("\n  Extraction des mots métier depuis la référence ELAN...")
    print("   - Ontologie SDIS (Regex)")
    print("   - Dictionnaire métier Excel (EntityRuler)")
    print("   - Détection automatique spaCy (NER)")
    
    # 1. Extraction sur la vérité terrain (ELAN)
    metier_words = extract_metier_words_with_spacy(ref_text)
    
    regex_count = len([m for m in metier_words if m['type'] == 'REGEX_SDIS'])
    ruler_count = len([m for m in metier_words if m['type'] == 'ENTITY_RULER'])
    spacy_count = len([m for m in metier_words if m['type'] == 'SPACY_NER'])
    
    print(f"   → Total: {len(metier_words)} mots/phrases métier")
    print(f"     - Regex SDIS: {regex_count}")
    print(f"     - Dictionnaire CISU: {ruler_count}")
    print(f"     - spaCy NER: {spacy_count}\n")
    
    # 2.Analyse contextuelle
    print("  Analyse contextuelle des mots métier...")
    analyses_contextes = analyser_tous_les_contextes(metier_words, ref_text)
    
    contextes_globaux = defaultdict(lambda: 0)
    faux_positifs = []
    
    for analyse in analyses_contextes:
        for contexte, data in analyse['contextes_detectes'].items():
            contextes_globaux[contexte] += 1
        
        if analyse['faux_positif_probable']:
            faux_positifs.append(analyse)
    
    print(f"   → Contextes détectés: {dict(contextes_globaux)}")
    print(f"   → Faux positifs probables: {len(faux_positifs)}\n")
    
    # 3. Calcul du Taux d'Erreur (WER) par entité
    results = {
        'correct': [],
        'substitutions': [],
        'deletions': [],
        'insertions': [],
        'contextes': analyses_contextes,
        'faux_positifs': faux_positifs
    }
    
    for metier in metier_words:
        error_type, forme_percue, score, info_spacy = analyze_whisper_error_with_spacy(metier, whisper_text)
        
        item = {
            'categorie': metier['categorie'],
            'mot_reference': metier['mot'],
            'type_detection': metier['type'],
            'source': metier['source']
        }
        
        # Ajout du contexte
        contexte_associe = next((a for a in analyses_contextes if a['mot'] == metier['mot']), None)
        if contexte_associe:
            item['contexte'] = contexte_associe['contextes_detectes']
            item['score_confidence'] = contexte_associe['score_confidence']
            item['interpretation'] = contexte_associe['interpretation']
            item['faux_positif_probable'] = contexte_associe['faux_positif_probable']
        
        if error_type == 'CORRECT':
            item['forme_percue'] = forme_percue
            results['correct'].append(item)
        elif error_type == 'SUBSTITUTION':
            item['forme_percue'] = forme_percue
            item['score_similarite'] = score
            if info_spacy:
                item['info_spacy'] = info_spacy
            results['substitutions'].append(item)
        else:
            results['deletions'].append(item)
    
    insertions = analyze_insertions_with_spacy(ref_text, whisper_text, metier_words)
    for ins in insertions:
        results['insertions'].append(ins)
    
    total_metier = len(metier_words)
    total_correct = len(results['correct'])
    total_substitutions = len(results['substitutions'])
    total_deletions = len(results['deletions'])
    
    wer_metier = (total_substitutions + total_deletions) / total_metier if total_metier > 0 else 0
    
    stats = {
        'total_metier': total_metier,
        'regex_count': regex_count,
        'ruler_count': ruler_count,
        'spacy_count': spacy_count,
        'correct': total_correct,
        'substitutions': total_substitutions,
        'deletions': total_deletions,
        'insertions': len(results['insertions']),
        'wer_metier': wer_metier,
        'precision_metier': (total_correct / total_metier * 100) if total_metier > 0 else 0,
        'contextes_globaux': dict(contextes_globaux),
        'faux_positifs_count': len(faux_positifs)
    }
    
    return results, stats, metier_words

# ============================================================
# RAPPORT COMPLET AVEC CONTEXTE
# ============================================================
def print_complete_report_avec_contexte(results, stats, metier_words):
	"""Affiche un résumé ergonomique dans la console pour l'utilisateur."""
    print("\n" + "="*80)
    print(" RAPPORT D'ANALYSE COMPLET AVEC CONTEXTE SPACY")
    print("="*80)
    
    print(f"\n  STATISTIQUES GLOBALES:")
    print(f"   Total concepts métier trouvés: {stats['total_metier']}")
    print(f"   ├─ Ontologie SDIS (Regex): {stats['regex_count']}")
    print(f"   ├─ Dictionnaire CISU (EntityRuler): {stats['ruler_count']}")
    print(f"   └─ Détection automatique (spaCy NER): {stats['spacy_count']}")
    print(f"\n    Corrects issus de Whisper : {stats['correct']} ({stats['precision_metier']:.1f}%)")
    print(f"    Substitutions: {stats['substitutions']}")
    print(f"    Délétions: {stats['deletions']}")
    print(f"    Insertions: {stats['insertions']}")
    print(f"\n    WER Métier: {stats['wer_metier']*100:.1f}%")
    
    # Contextes détectés
    print(f"\n\n  CONTEXTES DÉTECTÉS:")
    print("-" * 80)
    for contexte, count in sorted(stats['contextes_globaux'].items(), key=lambda x: -x[1]):
        print(f"   • {contexte}: {count} occurrence(s)")
    
    # Faux positifs probables
    if results['faux_positifs']:
        print(f"\n\n  ⚠️  FAUX POSITIFS PROBABLES ({len(results['faux_positifs'])}):")
        print("-" * 80)
        for fp in results['faux_positifs'][:10]:
            print(f"   ❌ Mot: '{fp['mot']}'")
            print(f"      Catégorie: {fp['categorie']}")
            print(f"      Raison: {', '.join(fp['raison_faux_positif'])}")
            print(f"      Contexte: {fp['interpretation']}")
            print()
    
    # Détail des concepts avec contexte
    print(f"\n\n  DÉTAIL DES CONCEPTS AVEC LEUR CONTEXTE:")
    print("-" * 80)
    
    for metier in metier_words[:20]:
        analyse = next((a for a in results['contextes'] if a['mot'] == metier['mot']), None)
        
        if analyse:
            confidence = "✅" if analyse['score_confidence'] > 0.5 else "⚠️" if analyse['score_confidence'] > 0.2 else "❌"
            faux_positif = " ⚠️ FP" if analyse['faux_positif_probable'] else ""
            
            print(f"\n   {confidence} [{metier['categorie'][:20]}] {metier['mot']}{faux_positif}")
            print(f"      Score confiance: {analyse['score_confidence']*100:.0f}%")
            print(f"      Contextes: {analyse['interpretation']}")
            
            for contexte, data in analyse['contextes_detectes'].items():
                print(f"        ↳ {contexte}: score={data['score']:.1f} mots={data['mots_contextuels'][:5]}")
    
    print("\n\n" + "="*80)
    print(" FIN DE L'ANALYSE CONTEXTUELLE")
    print("="*80)

# ============================================================
# SAUVEGARDE JSON
# ============================================================
def save_json_report(results, stats, input_filename):
	"""Sauvegarde les résultats complets pour une intégration Web/BI ultérieure."""
    nom_base = os.path.splitext(os.path.basename(input_filename))[0]
    output_file = f"rapport_wer_{nom_base}.json"
    
    # Nettoyer les contextes pour le JSON
    contextes_clean = []
    for c in results['contextes']:
        clean_c = {
            'mot': c['mot'],
            'categorie': c['categorie'],
            'type_detection': c['type_detection'],
            'source': c['source'],
            'score_confidence': c['score_confidence'],
            'interpretation': c['interpretation'],
            'faux_positif_probable': c['faux_positif_probable'],
            'raison_faux_positif': c['raison_faux_positif'],
            'contextes_detectes': {
                k: {'score': v['score'], 'mots_contextuels': v['mots_contextuels']}
                for k, v in c['contextes_detectes'].items()
            }
        }
        contextes_clean.append(clean_c)
    
    rapport = {
        'fichier_source': input_filename,
        'statistiques': stats,
        'details': {
            'correct': results['correct'],
            'substitutions': results['substitutions'],
            'deletions': results['deletions'],
            'insertions': results['insertions']
        },
        'analyse_contextuelle': {
            'contextes': contextes_clean,
            'faux_positifs': results['faux_positifs']
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n  Rapport JSON sauvegardé dans {output_file}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation complète avec spaCy NER et analyse contextuelle")
    parser.add_argument("--eaf", required=True, help="Chemin vers le fichier ELAN (.eaf)")
    parser.add_argument("--json", required=True, help="Chemin vers le fichier WhisperX (.json)")
    
    args = parser.parse_args()
    
    try:
        print(f"\n[CHARGEMENT] Analyse de {args.eaf} et {args.json}...")
        
        ref_data = load_elan_full_text(args.eaf)
        whisper_data = load_whisperx_full_text(args.json)
        
        print(f"   ✓ Référence ELAN: {len(ref_data['segments'])} segments, {len(ref_data['full_text'].split())} mots")
        print(f"   ✓ WhisperX: {len(whisper_data['words'])} mots")
        
        results, stats, metier_words = evaluate_metier_complete_avec_contexte(ref_data, whisper_data)
        print_complete_report_avec_contexte(results, stats, metier_words)
        save_json_report(results, stats, args.eaf)

        # Affichage détaillé en fin de script
        print("\n\n" + "="*80)
        print(" LISTE COMPLÈTE DES MOTS MÉTIER AVEC CONTEXTE")
        print("="*80)
        
        for i, metier in enumerate(metier_words, 1):
            analyse = next((a for a in results['contextes'] if a['mot'] == metier['mot']), None)
            if analyse:
                percu = any(item['mot_reference'] == metier['mot'] for item in results['correct'])
                statut = "✅ PERÇU" if percu else "❌ NON PERÇU"
                fp = " ⚠️ FP" if analyse['faux_positif_probable'] else ""
                conf = f"conf:{analyse['score_confidence']*100:.0f}%"
                print(f"{i:2}. [{metier['categorie'][:20]}] {metier['mot']:30} {statut} {conf} {fp}")
            else:
                print(f"{i:2}. [{metier['categorie'][:20]}] {metier['mot']:30} (sans contexte)")
        
        print("\n" + "="*80)
        print(" FIN DE L'ANALYSE")
        print("="*80)
        
    except Exception as e:
        print(f"\n  ERREUR: {e}")
        import traceback
        traceback.print_exc()
