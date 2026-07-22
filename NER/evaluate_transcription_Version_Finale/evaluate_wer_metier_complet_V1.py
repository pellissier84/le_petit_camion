import json
import re
import sys
import pympi
import unicodedata
import spacy
import argparse
import os
from collections import defaultdict
from difflib import SequenceMatcher

# nom du fichier : python evaluate_wer_metier_complet_V1.py
# evaluation et recherche des mots metiers transcription elan (eaf) versus whisper(json)
# commande : python evaluate_wer_metier_complet_V1.py --eaf "audio-1775031826.41778.eaf" --json "audio-1775031826.41778.json"
# audio-1775031826.41778
# audio-1775031968.41843
# audio-1775033540.32214
# calcul et visuel sur les evaluations de whisper
# utilise : EntityRuler de spacy avec patterns_sdis_cisu.jsonl (mots issus du CISU)
# utilise : Regex des entités nommées (mots métiers)
# utilise : gazetteer recuperation des communes du departement 31
# utilise : module de detection des entités nommés NER par spaCy

# 1. Chargement de l'IA (spaCy)
print("Chargement du modèle d'Intelligence Artificielle (spaCy)... patientez.")
nlp = spacy.load("fr_core_news_sm")

# ---  Ajout de l'EntityRuler avec le fichier JSONL ---
fichier_jsonl = "patterns_sdis_cisu.jsonl"
if os.path.exists(fichier_jsonl):
    # 'before="ner"' force spaCy à prioriser tes règles métier avant ses propres déductions
    ruler = nlp.add_pipe("entity_ruler", name="cisu_ruler", before="ner")
    ruler.from_disk(fichier_jsonl)
    print(f"[INFO] EntityRuler métier chargé avec succès depuis '{fichier_jsonl}'")
else:
    print(f"[AVERTISSEMENT] Le fichier '{fichier_jsonl}' est introuvable. L'analyse tournera sans le dictionnaire Excel.")
# -----------------------------------------------------------------

# 2. Chargement automatique du Gazetteer (Communes 31)
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
    return ''.join(
        c for c in unicodedata.normalize('NFD', texte)
        if unicodedata.category(c) != 'Mn'
    )

def commune_to_regex(nom_commune):
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

patterns_communes = list(set(commune_to_regex(c) for c in GAZETTEER_COMMUNES_31))
REGEX_COMMUNES = r"\b(" + "|".join(patterns_communes) + r")\b"

# 3. Ontologie Métier (Regex)
CONCEPTS_URGENCE = {
    "CONSCIENCE_NEURO": [
        r"(?i)\bne\s+(?:va|se\s+sent)\s+pas\s+bien\b", 
        r"(?i)\bne\s+r[eé]agit\s+pas\b", 
        r"(?i)\bne\s+bouge\s+(?:plus|pas)\b",
        r"(?i)\bne\s+(?:me\s+)?r[eé]pond(?:s|ent|ait)?\s+pas\b",
        r"(?i)\b(?:ne\s+(?:se\s+)?r[eé]veille\s+pas|dort|insensibles?)\b",
        # ---  Signes neuro pédiatriques (Hypotonie et pleurs) ---
        r"(?i)\b(?:b[eé]b[eé]s?|nourrissons?|enfants?|petits?)\b.{0,60}\b(?:moll?esses?|moux?|mous?|molles?|pleurs?\s+inconsolables?)\b",
        r"(?i)\b(?:moll?esses?|moux?|mous?|molles?|pleurs?\s+inconsolables?)\b.{0,60}\b(?:b[eé]b[eé]s?|nourrissons?|enfants?|petits?)\b",

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
        r"(?i)\b(?:l[èe]vres?\s+bleues?|peaux?\s+bleut[eé]e?s?|marbrures?(?:\s+de\s+la\s+peau)?|taches?\s+violac[eé]e?s?)\b", 
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
        r"(?i)\b(?:asthmes?|apn[eé]es?|insuffisances?\s+respiratoires?(?:\s+chroniques?)?|pendaisons?|pendus?|[eé]tranglements?)\b",
        r"(?i)\b(?:canules?|trach[eé]otomies?)\b",
        r"(?i)\b(?:son\s+(?:c[œo]ur|coeur)\s+ne\s+bat\s+pas|malades?\s+du\s+(?:c[œo]ur|coeur)|malades?\s+cardiaques?)\b",
        r"(?i)\b(?:douleurs?\s+dans\s+la\s+poitrine|(?:la\s+)?poitrine\s+(?:qui\s+(?:se\s+)?serre|dans\s+un\s+[eé]tau))\b",
        r"(?i)\b(?:a\s+d[eé]j[àa]\s+fait\s+un\s+)?infarctus\b",
        r"(?i)\b(?:le\s+)?pouls\s+(?:est\s+)?(?:filant|rapide|(?:ir)?r[eé]gulier|(?:im)?perceptible)\b",
        r"(?i)\b(?:maladies?\s+d[e'’\s]+(?:alzheimer|lewy|parkinson|charcot|crohn|lyme)|alzheimer|parkinson|lewy|corps\s+de\s+lewy|crohn|lyme)\b",
        r"(?i)\b(?:paludismes?|tuberculoses?|diab[èe]tes?|diab[eé]tiques?|cancers?|ob[eé]sit[eé]s?|sida|vih|thromboses?|emboll?ies?|ivg|m[eé]ningites?|coqueluches?|l[eé]gionn?ell?oses?|varioles?|chikungunya|dengue|hantavirus)\b",
        r"(?i)\b(?:(?:hyper|hypo)(?:glyc[eé]mies?|tensions?)|bronch(?:iol)?ites?|(?:cardio|art[eé]rio)pathies?|gastro[- ]?ent[eé]rites?)\b",
        r"(?i)\b(?:h[eé]patites?(?:\s+(?:[a-e]|virales?|et|,))*|interruptions?\s+volontaires?\s+d[e'’\s]+grossesses?|scl[eé]roses?\s+en\s+plaques?|fi[èe]vres?\s+jaunes?)\b", 
        r"(?i)\btemp[eé]ratures?(?:\s+corporelles?)?\s+[eé]lev[eé]e?s?\b",
        r"(?i)\b(?:fi[èe]vres?(?:\s+fortes?|\s+[eé]lev[eé]e?s?)?|fi[eé]vreu(?:x|ses?)|hyperthermies?)\b", 
        r"(?i)\b(?:ne\s+(?:s['’]\s*)?hydrate\s+pas|ne\s+boit\s+(?:pas|plus|pas\s+assez|que\s+tr[èe]s\s+peu|peu))\b",
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
        # Localisations anatomiques 
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
        # ---  Véhicules lourds et Transports de Matières Dangereuses (TMD) ---
        r"(?i)\b(?:poids\s+lourds?|ensembles?\s+routiers?)\b", 
        r"(?i)\bcamions?(?:\s+(?:petits?|gros?|citernes?|avec\s+remorques?))?\b",
        r"(?i)\b(?:avec\s+(?:un|une|des|les)?\s+)?(?:panneaux?|plaques?|[eé]tiquettes?)\s+(?:oranges?|dangers?|d[e'’]\s*produits?\s+dangereux)\b",
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
        r"(?i)\b(?:enfants?|b[eé]b[eé]s?|adultes?|p[èe]res?|m[èe]res?|papas?|mamans?|fils|filles?|gar[çc]ons?|adolescent(?:e?s?)?|voisin(?:e?s?)?|personnes?|patient(?:e?s?)?|nourrissons?|grand[- ]p[èe]res?|grand[- ]m[èe]res?|malades?)\b",
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
    "ACTIONS_SECOURS": [
        # --- RÉANIMATION  ---
        r"(?i)\b(?:j['’]ai\s+entrepris\s+une\s+r[eé]animation|j['’]ai\s+pos[eé]\s+un\s+dae|je\s+fais\s+un\s+massage\s+cardiaque)\b",

        # --- POSITION D'ATTENTE (PLS) ET PRISE EN CHARGE ---
        r"(?i)\b(?:je\s+l['’]ai\s+mis\s+(?:sur\s+le\s+c[oô]t[eé]|en\s+pls)|position\s+lat[eé]rale\s+de\s+s[eé]curit[eé])\b",
        r"(?i)\b(?:allonger?|allong[eé]e?s?)\s*(?:,?\s*et\s*)?\s*(?:couvrir|couvert(?:e?s?)?)\s*(?:,?\s*et\s*)?\s*(?:r[eé]conforter?|r[eé]confort[eé]e?s?)\b",

        # --- GESTION DES HÉMORRAGIES ET PLAIES ---
        r"(?i)\b(?:j['’]ai\s+(?:pos[eé]|mis)\s+un\s+(?:pansement(?:\s+compressif)?|garrot)|un\s+garrot|j['’]appuie\s+sur\s+(?:le\s+saignement|l['’]h[eé]morragie|la\s+plaie))\b",
        
        # --- LUTTE CONTRE L'INCENDIE ---
        r"(?i)\b(?:[eé]touffer?\s+(?:les\s+flammes|le\s+feu)\s+avec\s+une\s+couverture)\b",
        r"(?i)\b(?:extincteurs?\s+(?:appropri[eé]s?|[aà]\s+eau(?:\s+pulv[eé]ris[eé]e?)?|[aà]\s+mousse)|classes?\s+[aA][bB])\b",

        # --- SURVIE INCENDIE / CONFINEMENT ---
        r"(?i)\b(?:(?:se\s+)?confiner|confin[eé]e?s?|calfeutrer?\s+(?:les\s+)?portes?|se\s+manifester)\b",
        r"(?i)\b(?:respirer?\s+[aà]\s+travers\s+(?:un\s+)?linges?\s+mouill[eé]e?s?|respirer?\s+en\s+partie\s+basse|s['’]\s*accroupir?|accroupi(?:e?s?)?)\b"
    ],
    "ELEMENTS_PROTECTION": [
        # Dangers génériques et chutes
        r"(?i)\b(?:présences?\s+de\s+)?dangers?\b",
        r"(?i)\b(?:[eé]chelles?|objets?)\s+mena[çc]ants?\s+de\s+tomber\b",
        
        # Électricité et objets coupants
        r"(?i)\bfils?\s+[eé]lectriques?\b",
        r"(?i)\bobjets?\s+coupants?\b",
        
        # Machines et appareils en fonctionnement
        r"(?i)\b(?:appareils?|machines?)(?:\s+[eé]lectriques?)?\s+(?:en\s+marche|en\s+route|fonctionnant)\b",
        r"(?i)\b(?:[aà]\s+)?moteurs?\s+(?:fonctionnant|en\s+marche|en\s+route)\b",
        r"(?i)\b(?:tron[çc]onneuses?|scies?|tondeuses?|perceuses?|robots?\s+m[eé]nagers?)(?:\s+(?:en\s+route|en\s+marche|fonctionnant|allum[eé]e?s?))?\b",
        
        # Risques de circulation (Voie publique)
        r"(?i)\b(?:retir[eé]e?s?\s+(?:du\s+danger|de\s+la\s+route|de\s+la\s+pi[èe]ce\s+enfum[eé]e?s?)|j['’]ai\s+(?:pos[eé]\s+une\s+couverture|couvert(?:e|s)?)|(?:personnes?\s+)?au\s+milieu\s+d[e'’]\s*(?:la\s+)?(?:voies?\s+de\s+circulation|routes?|chauss[eé]es?))\b",
        # Risques de noyade / Milieu aquatique
        r"(?i)\b(?:[àa]\s+la\s+lisi[èe]re|au\s+bord)\s+d[e'’]\s*(?:un\s+|une\s+)?(?:plans?\s+d['’]eau|cours?\s+d['’]eau|mers?|rivi[èe]res?|fleuves?|can(?:al|aux))\b"
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

def normalize_text_for_detection(text):
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r"[^\w\s]", " ", text)
    
    disfluences = {
        "ah", "aie", "atchoum", "baf", "bah", "be", "ben", "bien", "bof",
        "bon ben", "bouh", "euh", "euf", "ha", "heu", "heueu", "he", "he bien",
        "hein", "hi", "ih", "hm", "hop", "hou", "hum", "hup", "la", "mah", 
        "menfin", "mmm", "mouais", "moui", "of", "oh", "ok", "okay", "ouah", 
        "ouais", "ouf", "ouille", "pff", "pouh", "snif", "tac", "toc", 
        "wahou", "yeah", "zut", "zou", "und"
    }
    
    for df in sorted(disfluences, key=len, reverse=True):
        pattern = r"\b" + re.escape(df) + r"\b"
        text = re.sub(pattern, " ", text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text_for_display(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_metier_words_with_spacy(text):
    """
    Extraction des mots métier avec TROIS méthodes :
    1. Regex (ontologie SDIS)
    2. EntityRuler (Dictionnaire métier CISU issu du JSONL)
    3. spaCy NER (détection automatique et rattrapage)
    """
    matches = []
    text_lower = text.lower()
    
    # --- PASSE 1 : Regex métier (SDIS) ---
    for categorie, patterns in CONCEPTS_URGENCE.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
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
    
    # --- PASSE 2 & 3 : EntityRuler + spaCy NER ---
    doc = nlp(text)
    for ent in doc.ents:
        # Catégorisation des entités selon la source
        if ent.label_.startswith("CISU_"):  # C'est un pattern de notre fichier JSONL !
            categorie_ner = ent.label_
            type_detection = 'ENTITY_RULER'
            source_detection = f"Dictionnaire CISU Excel"
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
            continue
        
        # Éviter les doublons avec les regex
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
    
    # Supprimer les doublons exacts
    matches_unis = []
    vus = set()
    for m in matches:
        cle = (m['categorie'], m['mot'].lower())
        if cle not in vus:
            vus.add(cle)
            matches_unis.append(m)
    
    return matches_unis

def load_whisperx_full_text(filepath):
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

def analyze_whisper_error_with_spacy(metier_item, whisper_text):
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
    
    return 'DELETION', None, 0, None

def analyze_insertions_with_spacy(ref_text, whisper_text, metier_words):
    ref_norm = set(normalize_text_for_detection(ref_text).split())
    whisper_norm = set(normalize_text_for_detection(whisper_text).split())
    
    metier_mots_ref = set()
    for m in metier_words:
        for mot in normalize_text_for_detection(m['mot']).split():
            metier_mots_ref.add(mot)
    
    insertions = whisper_norm - ref_norm
    
    stopwords = {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 'donc', 'car', 
                 'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'dans', 
                 'par', 'sur', 'pour', 'avec', 'sans', 'chez', 'est', 'sont', 'avait', 'avais'}
    
    insertions_significatives = []
    for mot in insertions:
        if len(mot) > 3 and mot not in stopwords and mot not in metier_mots_ref:
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

def evaluate_metier_complete(ref_data, whisper_data):
    ref_text = ref_data['full_text']
    whisper_text = whisper_data['full_text']
    
    print("\n  Extraction des mots métier depuis la référence ELAN...")
    print("   - Ontologie SDIS (Regex)")
    print("   - Dictionnaire métier Excel (EntityRuler)")
    print("   - Détection automatique spaCy (NER)")
    
    metier_words = extract_metier_words_with_spacy(ref_text)
    
    regex_count = len([m for m in metier_words if m['type'] == 'REGEX_SDIS'])
    ruler_count = len([m for m in metier_words if m['type'] == 'ENTITY_RULER'])
    spacy_count = len([m for m in metier_words if m['type'] == 'SPACY_NER'])
    
    print(f"   → Total: {len(metier_words)} mots/phrases métier")
    print(f"     - Regex SDIS: {regex_count}")
    print(f"     - Dictionnaire CISU: {ruler_count}")
    print(f"     - spaCy NER: {spacy_count}\n")
    
    results = {
        'correct': [],
        'substitutions': [],
        'deletions': [],
        'insertions': []
    }
    
    for metier in metier_words:
        error_type, forme_percue, score, info_spacy = analyze_whisper_error_with_spacy(metier, whisper_text)
        
        item = {
            'categorie': metier['categorie'],
            'mot_reference': metier['mot'],
            'type_detection': metier['type'],
            'source': metier['source']
        }
        
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
        'precision_metier': (total_correct / total_metier * 100) if total_metier > 0 else 0
    }
    
    return results, stats, metier_words

def print_complete_report(results, stats, metier_words):
    print("\n" + "="*80)
    print(" RAPPORT D'ANALYSE COMPLET AVEC SPACY NER ET DICTIONNAIRE CISU")
    print("="*80)
    
    print(f"\n  STATISTIQUES GLOBALES:")
    print(f"   Total concepts métier trouvés: {stats['total_metier']}")
    print(f"   ├─ Ontologie SDIS (Regex): {stats['regex_count']}")
    print(f"   ├─ Dictionnaire CISU (EntityRuler): {stats['ruler_count']}")
    print(f"   └─ Détection automatique (spaCy NER): {stats['spacy_count']}")
    print(f"\n    Corrects issu de Whisper : {stats['correct']} ({stats['precision_metier']:.1f}%)")
    print(f"    Substitutions: {stats['substitutions']}")
    print(f"    Délétions: {stats['deletions']}")
    print(f"    Insertions: {stats['insertions']}")
    print(f"\n    WER Métier: {stats['wer_metier']*100:.1f}%")
    
    if results['correct']:
        print(f"\n\n  CONCEPTS CORRECTEMENT PERÇUS par Whisper ({len(results['correct'])}):")
        print("-" * 80)
        for item in results['correct'][:15]:
            print(f"   [{item['categorie']}] '{item['mot_reference']}' → '{item['forme_percue']}'")
            print(f"   └─ Détecté par: {item['source']}")
    
    if results['substitutions']:
        print(f"\n\n  SUBSTITUTIONS de Whisper ({len(results['substitutions'])}):")
        print("-" * 80)
        for item in results['substitutions'][:10]:
            print(f"   [{item['categorie']}] '{item['mot_reference']}'")
            print(f"   → Remplacé par: '{item['forme_percue']}' (sim={item['score_similarite']*100:.0f}%)")
            print(f"   └─ Détecté par: {item['source']}")
    
    if results['deletions']:
        print(f"\n\n  CONCEPTS MANQUANTS ratés par whisper({len(results['deletions'])}):")
        print("-" * 80)
        for item in results['deletions'][:10]:
            print(f"   [{item['categorie']}] '{item['mot_reference']}'")
            print(f"   └─ Détecté par: {item['source']}")
    
    if results['insertions']:
        print(f"\n\n  HALLUCINATIONS DÉTECTÉES ou insertions de Whisper({len(results['insertions'])}):")
        print("-" * 80)
        spacy_insertions = [i for i in results['insertions'] if i.get('type_spacy') and i['type_spacy'] != 'mot_inconnu']
        if spacy_insertions:
            print(f"\n    Entités nommées hallucinées (spaCy):")
            for ins in spacy_insertions[:10]:
                print(f"   - '{ins['mot']}' (type: {ins['type_spacy']})")
    
    print("\n\n" + "="*80)
    print(" SYNTHÈSE PAR TYPE DE CONCEPT")
    print("="*80)
    
    types_performance = defaultdict(lambda: {'total': 0, 'correct': 0, 'by_source': defaultdict(int)})
    
    for item in results['correct']:
        if 'AUTO' in item['categorie']:
            racine = item['categorie'].split('_AUTO')[0]
        else:
            racine = item['categorie']
        types_performance[racine]['total'] += 1
        types_performance[racine]['correct'] += 1
        types_performance[racine]['by_source'][item['source']] += 1
    
    for item in results['substitutions']:
        if 'AUTO' in item['categorie']:
            racine = item['categorie'].split('_AUTO')[0]
        else:
            racine = item['categorie']
        types_performance[racine]['total'] += 1
    
    for item in results['deletions']:
        if 'AUTO' in item['categorie']:
            racine = item['categorie'].split('_AUTO')[0]
        else:
            racine = item['categorie']
        types_performance[racine]['total'] += 1
    
    for typ, data in sorted(types_performance.items()):
        taux = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        barre = "█" * int(taux / 5) + "░" * (20 - int(taux / 5))
        print(f"\n{typ:25} [{barre}] {taux:.0f}% ({data['correct']}/{data['total']})")
        if data['by_source']:
            print(f"   └─ Sources: {dict(data['by_source'])}")

def save_json_report(results, stats, input_filename):
    # On extrait le nom de base : 'audio-1775031826.41778.eaf' -> 'audio-1775031826.41778'
    nom_base = os.path.splitext(os.path.basename(input_filename))[0]
    
    # On définit le nom de sortie normalisé
    output_file = f"rapport_wer_{nom_base}.json"
    
    rapport = {
        'fichier_source': input_filename, # Optionnel : ajoute une trace dans le JSON
        'statistiques': stats,
        'details': {
            'correct': results['correct'],
            'substitutions': results['substitutions'],
            'deletions': results['deletions'],
            'insertions': results['insertions']
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n  Rapport JSON sauvegardé dans {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation complète avec spaCy NER et EntityRuler")
    parser.add_argument("--eaf", required=True, help="Chemin vers le fichier ELAN (.eaf)")
    parser.add_argument("--json", required=True, help="Chemin vers le fichier WhisperX (.json)")
    
    args = parser.parse_args()
    
    try:
        print(f"\n[CHARGEMENT] Analyse de {args.eaf} et {args.json}...")
        
        ref_data = load_elan_full_text(args.eaf)
        whisper_data = load_whisperx_full_text(args.json)
        
        print(f"   ✓ Référence ELAN: {len(ref_data['segments'])} segments, {len(ref_data['full_text'].split())} mots")
        print(f"   ✓ WhisperX: {len(whisper_data['words'])} mots")
        
        results, stats, metier_words = evaluate_metier_complete(ref_data, whisper_data)
        print_complete_report(results, stats, metier_words)
        save_json_report(results, stats, args.eaf)

        print("\n\n" + "="*80)
        print(" LISTE COMPLÈTE DES MOTS MÉTIER IDENTIFIÉS")
        print("="*80)
        
        for i, metier in enumerate(metier_words, 1):
            percu = any(item['mot_reference'] == metier['mot'] for item in results['correct'])
            statut = "✅ PERÇU" if percu else "❌ NON PERÇU"
            print(f"{i:2}. [{metier['categorie']:25}] {metier['mot']:35} {statut}")
        
        print("\n" + "="*80)
        print(" FIN DE L'ANALYSE")
        print("="*80)
        
    except Exception as e:
        print(f"\n  ERREUR: {e}")
        import traceback
        traceback.print_exc()
