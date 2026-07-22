import json
import re
import os
import unicodedata
import spacy
from collections import defaultdict

# recherche des mots metiers
# a partir du json whisper et json pyannote
# par rapport aux audio wav
# audio-1775031968.41843
# audio-1775033540.32214
# audio-1775031826.41778
# recherche par regex, ner ,EntityRuler (patterns_sdis_CISU), gazetteer communes
# affichage résultat sur ecran et sous forme de json

"""
Script d'extraction d'entités et de mots métiers pour les appels d'urgence (CISU/SDIS).
Combine une ontologie par expressions régulières (Regex) et la reconnaissance 
d'entités nommées (NER) via spaCy pour repérer des concepts critiques dans des 
transcriptions WhisperX, puis attribue ces concepts aux locuteurs via Pyannote.
"""

# 1. Chargement de l'IA (spaCy)
print("Chargement du modèle d'Intelligence Artificielle (spaCy)... patientez.")
nlp = spacy.load("fr_core_news_sm")

fichier_jsonl = "patterns_sdis_cisu.jsonl"

if os.path.exists(fichier_jsonl):
    # Ajout d'un EntityRuler avant le NER standard de spaCy
    # Permet de forcer la détection de schémas spécifiques CISU avant l'analyse par défaut
    ruler = nlp.add_pipe("entity_ruler", name="cisu_ruler", before="ner")
    ruler.from_disk(fichier_jsonl)
    print(f"[INFO] EntityRuler métier chargé avec succès depuis '{fichier_jsonl}'")
else:
    print(f"[AVERTISSEMENT] Le fichier '{fichier_jsonl}' est introuvable.")

# 2. Chargement du Gazetteer (Communes 31)
def charger_gazetteer(chemin_fichier):
    """
    Charge une liste de communes depuis un fichier GeoJSON/JSON.
    
    Args:
        chemin_fichier (str): Chemin vers le fichier contenant les données géographiques.
        
    Returns:
        list: Liste dédoublonnée des noms de communes en minuscules et sans tirets.
    """
    villes = []
    if os.path.exists(chemin_fichier):
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            if chemin_fichier.endswith('.json'):
                donnees = json.load(f)
                for feature in donnees.get('features', []):
                    ville = feature.get('properties', {}).get('nom_commune', '')
                    if ville:
                        # Normalisation basique : minuscules et remplacement des tirets
                        villes.append(ville.strip().lower().replace("-", " "))
        print(f"[INFO] {len(set(villes))} communes chargées.")
    return list(set(villes))

GAZETTEER_COMMUNES_31 = charger_gazetteer("communes-haute-garonne.geojson.json")

def enlever_accents(texte):
    """Supprime les accents d'une chaîne de caractères via normalisation Unicode."""
    return ''.join(c for c in unicodedata.normalize('NFD', texte) if unicodedata.category(c) != 'Mn')

def commune_to_regex(nom_commune):
    """
    Transforme un nom de commune en une expression régulière tolérante aux variations 
    orthographiques fréquentes (accents manquants, tirets vs espaces).
    """
    base = enlever_accents(nom_commune.lower().strip())
    # Rend les espaces et tirets interchangeables
    base = re.sub(r'[\s\-]+', r'[\\s-]+', base)
    # Mapping pour tolérer la présence ou l'absence d'accents
    mapping = {'e': '[eéèêë]', 'a': '[aàâä]', 'o': '[oôö]', 'i': '[iîï]', 'u': '[uùûü]', 'c': '[cç]'}
    return "".join(mapping.get(c, c) for c in base)

# Création d'une méga-regex combinant toutes les communes du gazetteer
patterns_communes = list(set(commune_to_regex(c) for c in GAZETTEER_COMMUNES_31))
REGEX_COMMUNES = r"\b(" + "|".join(patterns_communes) + r")\b" if patterns_communes else r"\b\b"

# 3. Ontologie Métier (Regex)
# Dictionnaire regroupant les expressions régulières par catégorie d'urgence.
# L'utilisation de (?i) rend la recherche insensible à la casse.
# Les \b s'assurent de matcher des mots entiers pour éviter les faux positifs.
CONCEPTS_URGENCE = {
    "CONSCIENCE_NEURO": [
        r"ne\s+va\s+pas\s+bien", r"ne\s+se\s+sent\s+pas\s+bien", r"ne\s+réagit\s+pas", 
        r"ne\s+bouge\s+(plus|pas)",
        r"ne\s+(me\s+)?r[eé]pond(s|ent|ait)?\s+pas",
        # Capte: allongé/couché/étendu, avec ou sans "au sol" ou "par terre"
        r"(?i)\b(?:allong[eé]e?s?|couch[eé]e?s?|[eé]tendue?s?)(?:\s+(?:au\s+sol|par\s+terre))?\b",
    
        # Capte: tombé (et en bonus, on anticipe "tombé au sol" ou "tombé par terre")
        r"(?i)\btomb[eé]e?s?(?:\s+(?:au\s+sol|par\s+terre))?\b",
    
        # Mots isolés
        r"(?i)\binertes?\b",
        r"(?i)\ballit[eé]e?s?\b",
        r"perte\s+de\s+(connaissance|conscience)", 
        r"tremble[s]?", r"tremblement[s]?", r"crise\s+de\s+(tremblements?|spasmophilie|tétanie)", 
        r"crise\s+d[’']épilepsie", r"malaise", r"paralys(é|ée|ie)", r"yeux\s+fixes",
        r"troubles?\s+sensoriels?", r"convulsions?", r"troubles?\s+d[’']élocution",
        r"pupilles\s+dilatées?", r"vision\s+troublée?", r"troubles?\s+de\s+la\s+vision",
        r"perte\s+d[’']équilibre", r"vomissements?", r"vertiges?", r"mal\s+de\s+tête", r"maux\s+de\s+tête",
        r"sous\s+emprise\s+de\s+(drogues?|alcool|stupéfiants?|médicaments?)", r"sous\s+(traitement|médicaments?)",
        r"électrocutée?", r"électrocution", 
        r"(?i)\b(?:ingestions?|inhalations?|injections?)\s+(?:de\s+|d['’]\s*|des\s+)(?:produits?\s+(?:toxiques?|chimiques?|pharmaceutiques?|caustiques?)|m[eé]dicaments?|acides?|caustiques?|d[eé]tergents?)\b",
        r"évanouissement", r"évanouie?", r"sueurs?", r"pâleur|paleur", r"blancheur|blanc|blanche", r"livide"
    ],
    "DETRESSE_VITALE": [
        # --- SAIGNEMENTS ET PLAIES ---
        r"(?i)\b(?:flaques?\s+de\s+sang|baigne\s+dans\s+son\s+sang|se\s+vide\s+de\s+son\s+sang|saigne\s+abondamment|sang)\b",
        r"(?i)\bsaignements?(?:\s+(?:des\s+oreilles|par\s+la\s+bouche|abondants?|en\s+jet|en\s+saccades?|sous\s+pression))?\b",
        r"(?i)\b(?:coupures?|entailles?|taillader?(?:\s+les\s+veines)?|plaies?\s+profondes?|objet\s+dans\s+la\s+plaie)\b",
    
        # --- RESPIRATION ---
        r"(?i)\b(?:difficult[eé]s?\s+[àa]\s+respirer|ne\s+respire\s+pas|asphyxies?|toux|s['’][eé]touffe|respire\s+(?:avec\s+difficult[eé]|avec\s+bruit|rapidement|bruyamment))\b",
        r"cr[ée]pitements?",
        
        # --- CARDIAQUE ET POULS (Nouveautés incluses) ---
        r"(?i)\b(?:son\s+(?:cœur|coeur)\s+ne\s+bat\s+pas|malades?\s+du\s+(?:cœur|coeur)|malades?\s+cardiaques?|douleurs?\s+dans\s+la\s+poitrine)\b",
        r"(?i)\b(?:a\s+d[eé]j[àa]\s+fait\s+un\s+)?infarctus\b",
        r"(?i)\b(?:le\s+)?pouls\s+(?:est\s+)?(?:filant|rapide|(?:ir)?r[eé]gulier|(?:im)?perceptible)\b",
    
        # --- PATHOLOGIES DIVERSES ---
        r"(?i)\b(?:br[ûu]lures?|an[eé]mi[eé]e?s?|h[eé]morragies?|m[eé]trorr ?agies?|œd[èe]mes?|oed[èe]mes?|diab[èe]tes?|diab[eé]tiques?)\b",
        r"(?i)\b(?:maladies?\s+d[e'’\s]+(?:alzheimer|lewy|parkinson|charcot)|alzheimer|parkinson|lewy|corps\s+de\s+lewy)\b",
    
        # --- ACTIONS DE SECOURS (Nouveautés incluses) ---
        r"(?i)\b(?:j['’]ai\s+entrepris\s+une\s+r[eé]animation|j['’]ai\s+pos[eé]\s+un\s+dae|je\s+fais\s+un\s+massage\s+cardiaque)\b"
    ],
    "TRAUMATISME": [
        r"chute( de sa hauteur| des escaliers| d[’']une échelle| sur la tête)?",
        r"déformation( du visage| de la bouche)?", r"douleur( au dos)?",
        r"cassé", r"déplacé", r"amputé", r"coupé",
        r"fortes douleurs?|douleurs? vives?", r"incapable (de|à) bouger", r"incapacité [àa] bouger",
        r"amputation", r"luxation", r"déboîte(r|ment)? l[’']articulation"
    ],
   "INCENDIE_GAZ": [
        # --- FUMÉE ---
        # Capte : fumée(s), panache de fumée, fumée qui pique, fumée s'échappe...
        r"(?i)\b(?:panaches?\s+de\s+)?fum[eé]es?(?:\s+(?:opaques?|noires?|grises?|blanches?|irrespirables?|qui\s+piquent?\s+les\s+yeux|s['’][eé]chappent?))?\b",

        # --- FEU & VOCABULAIRE MATÉRIEL ---
        r"(?i)\b(?:incendies?|feux?|boules?\s+de\s+feu|foyers?|brasiers?|torch[eè]res?|lueurs?|[eé]tincelles?)\b",
    
        # --- FLAMMES & CHALEUR ---
        r"(?i)\b(?:flammes?|en\s+flammes?|cr[eé]pitements?(?:\s+des\s+flammes)?|chaleurs?\s+intenses?)\b",

        # --- GAZ & ODEURS (Correction de "odeur nauséabonde" incluse) ---
        r"(?i)\bodeurs?\s+(?:de\s+(?:gaz|br[ûu]l[eé])|naus[eé]abondes?)\b",
        r"(?i)\bgaz\s+(?:toxiques?|irritants?)\b",
        r"(?i)\bfuites?\s+de\s+gaz(?:\s+enflamm[eé]e?s?)?\b",

        # --- VERBES ET ACTIONS (brûle, flambe, s'embrase...) ---
        # Capte : brûle, brûler, brule, en train de brûler
        r"(?i)\b(?:(?:en\s+train\s+de\s+)?br[ûu]le(?:r|nt|s)?)\b",
        # Capte : ça flambe, ca flambe, cà flambe
        r"(?i)\b(?:[cç][aà]\s+flambe|allumer|enflammer|incendier|embraser)\b",
        # Capte : embrasé, s'est embrasé, s'est embrasée...
        r"(?i)\b(?:s['’]est\s+)?embras[eé]e?s?\b",

        # --- CONSÉQUENCES ---
        r"(?i)\b(?:explosions?|intoxications?)\b"
    ],
    "ACCIDENT_ROUTE": [
        # Types d'accidents
        r"(?i)\baccidents?(?:\s+(?:routiers?|graves?|simples?|l[eé]gers?))?\b",
        r"(?i)\bcarambolages?\b",  r"(?i)\btonneaux?\b",
    
        # Types de chocs (Gère "choc frontal", "chocs frontaux", "latéral/latéraux", etc.)
        r"(?i)\bchocs?\s+(?:front(?:al|aux)|lat[eé]r(?:al|aux)|(?:par\s+)?l['’]arri[eè]re)\b",
    
        # État de la voiture (Gère les accords, les accents et l'adverbe optionnel)
        r"(?i)\bvoitures?\s+(?:compl[eè]tement\s+)?(?:d[eé]form[eé]e?s?|explos[eé]e?s?|d[eé]truite?s?|[eé]cras[eé]e?s?)\b",

        r"éjecté", r"coincé", r"personne (éjectée|coincée|sortie)",
        r"(grande|petite) vitesse", r"cinétique importante",
        r"poids lourd", r"trottinette( électrique)?", r"voiture[s]?", r"camion( petit| gros)?", 
        r"bus", r"tracteur", r"scooter", r"vélo", r"cycliste", r"piéton"
    ],
    "LOCALISATION_SPECIFIQUE": [
        # --- 1. INFRASTRUCTURES GÉNÉRALES ---
        r"(?i)\b(?:en\s+ville|(?:aux?|un|des|les?)\s+(?:ronds?[- ]points?|croisements?))\b",
        r"(?i)\b(?:autoroutes?|voies?(?:\s+rapides?)?|ponts?|p[eé]ages?|tunnels?)\b",

        # --- 2. GÉOMÉTRIE DE LA ROUTE ---
        r"(?i)\bdans\s+(?:un|des|les?)\s+virages?\b",
        r"(?i)\bdans\s+(?:une|des|les?)\s+courbes?\b",
        r"(?i)\ben\s+lignes?\s+droites?\b",

        # --- 3. AMÉNAGEMENTS ET SÉCURITÉ  ---
        # Capte: terre-plein, terre-plein central, terre plein...
        r"(?i)\bterres?[- ]pleins?(?:\s+centra(?:l|ux))?\b",
        # Capte: barrière(s) de sécurité, glissière(s), rambarde(s)...
        r"(?i)\b(?:barri[èe]res?|glissi[èe]res?|rambardes?)(?:\s+de\s+s[eé]curit[eé])?\b",
        # Capte: trottoir(s)
        r"(?i)\b(?:sur\s+le\s+)?trottoirs?\b",

        # --- 4. RELIEF ET HORS-ROUTE ---
        r"(?i)\bdans\s+(?:un|le|les?|des)\s+(?:foss[eé]s?|pr[eé]cipices?)\b",
        r"(?i)\b(?:en\s+contre[- ]bas|abrupte?s?|d[eé]nivel[eé]s?\s+important(?:e?s?)?)\b",

        # --- 5. OBSTACLES & CINÉMATIQUE ---
        # Capte: a quitté, à quitté, a quitter la route
        r"(?i)\b[aà]\s+quitt[eé](?:r|s?|es?)?\s+la\s+route\b",
        # Capte: contre un arbre, contre des arbres
        r"(?i)\bcontre\s+(?:un|des|les?)\s+arbres?\b",
        # Capte: un rocher, des rochers, contre un rocher
        r"(?i)\b(?:contre\s+)?(?:un|des|les?)\s+rochers?\b",
        # Capte: contre un mur, muret, poteau, lampadaire, pylône 
        r"(?i)\b(?:contre\s+)?(?:un|des|les?)\s+(?:murs?|murets?|poteaux?|lampadaires?|pyl[ôo]nes?)\b",

        # --- 6. BÂTIMENTS & ÉTAGES ---
        r"(?i)\b(?:rez[- ]de[- ]chauss[eé]e|rdc|niveau\s+(?:z[eé]ro|0))\b",
        r"(?i)\b[àa]\s+l['’][eé]tage\b",
        # Étages (ex: 1er étage, deuxième étage, étage 2...)
        r"(?i)\b(?:\d{1,2}(?:er|ème|eme|e)?|premier|deuxi[èe]me|troisi[èe]me|quatri[èe]me|cinqui[èe]me|sixi[èe]me|septi[èe]me|huiti[èe]me|neuvi[èe]me|dixi[èe]me|onzi[èe]me|douzi[èe]me|un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze)\s+[eé]tages?\b",
        r"(?i)\b[eé]tages?\s+(?:\d{1,2}|un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\b",
        # Numéro d'appartement (ex: appart numéro 254 b)
        r"(?i)\bappart(?:ement)?\s+(?:num[eé]ro\s+|n[°o]\s+)?(\d{1,4}(?:\s*[a-z]+)?|[a-z])\b",

        # --- 7. ROUTES & VOIES (Corrigé pour éviter les faux positifs) ---
        # Règle A : Mots complets + Nombres (ex: "départementale six", "RN20")
        r"(?i)\b(?:autoroute|d[eé]partementale|nationale|rn|rd)\s*[-]?\s*(?:\d+|(?:(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|et)\s*[-]?\s*)+)\b",
        # Règle B : Lettres isolées + Chiffres uniquement (ex: "D 622", "a7", "A 25")
        r"\b(?:(?i:[dnl]\s*\d{1,4}|a\d{1,4})|A\s+\d{1,4})\b",
        # Règle C : Routes D et N + Nombres en toutes lettres (ex: "d six cent", exclut le "a" pour éviter "a un")
        r"(?i)\b(?:d|n)\s+(?:(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|et)\s*[-]?\s*)+\b",

        # --- 8. ADRESSES CLASSIQUES ---
        # Capte: numéro (chiffres ou lettres) + type de voie + nom de la rue
        r"(?i)\b(?:(?:(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|et)(?:[\s-]+(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|et))*|\d{1,4})(?:\s*(?:bis|ter|quater|[a-z]))?\s*,?\s+)?(?:r[eé]sidence|chemin|rue|avenue|impasse|place|b[aâ]timent|boulevard|all[eé]e|route|voie|cours|quai|lieux?[- ]dits?|quartiers?)(?:(?:\s+|-)(?:(?:de\s+la|de\s+l['’]|des|du|de|la|le|les|l['’]|d['’])\s+)?(?!(?:et|il|elle|je|tu|on|nous|vous|ils|elles|dans|devant|derri[èe]re|autour|avec|pour|qui|que|quoi|est|sont|ont|[cç]a?|ce|cette|ces|ceux|celles?|tout|toute?s?|tr[èe]s|mais|donc|car|puis|ensuite|voil[àa]|l[àa]|par|ici|vite|y|en|un|une)\b)[a-zà-âçéèêëîïôûùü0-9]+){0,4}\b",
        # Mots isolés pour rues et bâtiments (Sans numéro)
        r"(?i)\b(?:petites?\s+rues?|rues?\s+[eé]troites?|chemins?\s+de\s+terre|impasses?|tunnels?|ruelles?|passages?|b[aâ]timents?|maisons?)\b",

        # --- 9. HYDROGRAPHIE & POINTS D'EAU ---
        r"(?i)\b(?:torrents?|rivi[èe]res?|lacs?|fleuves?|trous?|puits|foss[eé]s?\s+profonds?|ruisseaux?|cours?\s+d['’]eau)\b"
    ],
    "COMMUNES_31": [
        REGEX_COMMUNES
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
        r"enfant", r"bébé", r"adulte", r"père", r"mère", r"fils", r"fille", 
        r"voisin[e]?", r"personne", r"patient[e]?",
        r"nourrisson[s]?", r"grand[-]père", r"grand[-]mère", r"personne[s]? âgée[s]?",
        r"personne handicapée", r"avec un handicap", r"en fauteuil roulant",
        r"malade[s]?", r"piéton|pieton", r"marcheur[s]?", r"randonneur[s]?", r"cycliste[s]?", r"motard[s]?"
    ],
    "AGE_VICTIME": [
        r"(?i)\b(\d+|(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|et)(?:[\s-]+(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|et))*)\s+(ans?|mois|jours?|semaines?)\b"
    ],
    "MESURES": [
        # distance
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:m[eè]tres?|centim[eè]tres?|millim[eè]tres?|kilom[eè]tres?|km|cm|mm|m)\b",
        r"(?i)\b(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|dizaine|centaine|et|de)(?:[\s-]+(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|milles?|dizaine|centaine|et|de))*\s+(?:m[eè]tres?|centim[eè]tres?|millim[eè]tres?|kilom[eè]tres?|km|cm|mm|m)\b",
        # numeros de telephone
        r"\b0[1-9](?:[\s.-]?\d{2}){4}\b",
        r"(?:\b0|\+33\s?)[1-9](?:[\s.-]?\d{2}){4}\b",
        # code postal français
        r"\b\d{5}\b"
    ],
    "DEMANDE_MOYENS": [
        r"demande d[’']ambulance", r"médecin|docteur", r"infirmi[eèé]re?s?", r"demande de secours",
        r"camion de feu", r"camion pour combattre l[’']incendie", r"lutter contre l[’']incendie",
        r"(gros|petit) camion", r"grande échelle", r"voiture de pompier[s]?", r"véhicule de secours"
    ]
}

def extract_metier_words_with_spacy(text):
    """
    Analyse un texte brut pour en extraire les mots métiers via une approche hybride.
    Combine les Regex (Ontologie SDIS) et le NLP (spaCy NER + EntityRuler).
    
    Args:
        text (str): La transcription texte complète à analyser.
        
    Returns:
        list: Liste de dictionnaires contenant les métadonnées de chaque entité trouvée.
    """
    matches = []
    text_lower = text.lower()
    
    # PASSE 1 : Détection par Expressions Régulières (Regex) ---
    for categorie, patterns in CONCEPTS_URGENCE.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                matches.append({
                    'categorie': categorie,
                    'mot': match.group(0).strip(),
                    'char_start': match.start(),
                    'char_end': match.end(),
                    'type': 'REGEX_SDIS',
                    'source': 'Ontologie SDIS'
                })
    
    # PASSE 2 & 3 : Détection NLP (EntityRuler + spaCy NER) ---
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_.startswith("CISU_"):
            cat = ent.label_
            typ, src = 'ENTITY_RULER', "Dictionnaire CISU"
        elif ent.label_ in ["LOC", "ORG", "PER", "PERSON", "DATE", "TIME", "CARDINAL"]:
            cat = f"AUTO_{ent.label_}"
            typ, src = 'SPACY_NER', f"spaCy ({ent.label_})"
        else:
            continue
            
        texte_entite = ent.text.strip()
        if len(texte_entite) < 3: continue # Ignorer les mots trop courts
        
        # Eviter doublons avec regex
        # Filtre anti-doublon : on vérifie si la Regex n'a pas déjà capturé ce mot
        if not any(existing['mot'] == texte_entite.lower() for existing in matches):
            matches.append({
                'categorie': cat,
                'mot': texte_entite,
                'char_start': ent.start_char,
                'char_end': ent.end_char,
                'type': typ,
                'source': src
            })
            
    # --- Dédoublonnage final ---
    # Évite d'avoir la même entité détectée plusieurs fois exactement au même endroit
    vus = set()
    matches_unis = []
    for m in matches:
        cle = (m['categorie'], m['char_start'], m['char_end'])
        if cle not in vus:
            vus.add(cle)
            matches_unis.append(m)
            
    return matches_unis

# GESTION DES FICHIERS AUDIO (WhisperX & Pyannote)
# ==========================================

def load_whisperx_and_build_offsets(filepath):
    """
    Charge un JSON issu de WhisperX et reconstruit le texte complet
    tout en gardant la trace des index de caractères (offsets) pour chaque mot.
    Cela permet de lier un caractère précis du texte à un timestamp audio.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tous_les_mots = []
    for segment in data.get('segments', []):
        for word_info in segment.get('words', []):
            if 'word' in word_info:
                tous_les_mots.append({
                    'word': word_info['word'],
                    'start': word_info.get('start', 0.0),
                    'end': word_info.get('end', 0.0)
                })
    
    # Sécurité : s'assurer que les mots sont bien dans l'ordre chronologique
    tous_les_mots.sort(key=lambda x: x['start'])
    
    texte_complet = ""
    for w in tous_les_mots:
        w['char_start'] = len(texte_complet)
        texte_complet += w['word']
        w['char_end'] = len(texte_complet)
        texte_complet += " " # Maintien d'un espace vital pour la précision des index
        
    return texte_complet.strip(), tous_les_mots

def load_pyannote_diarization(filepath):
    """Charge le fichier JSON contenant la segmentation Pyannote avec gestion d'erreurs."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERREUR CRITIQUE] Le fichier {filepath} n'est pas un JSON valide.")
        print(f"Détails de l'erreur : {e}")
        return None
    except FileNotFoundError:
        print(f"[ERREUR] Fichier introuvable : {filepath}")
        return None

def get_speaker_for_time(start_time, end_time, diarization_data, audio_filename):
    """
    Identifie le locuteur actif sur une plage temporelle donnée.
    Utilise une logique de calcul de chevauchement (overlap) : si le mot chevauche 
    plusieurs segments de locuteurs, il est attribué à celui qui occupe la plus grande part.
    """
    segments = []
    
    if isinstance(diarization_data, dict) and "files" in diarization_data:
        for audio_data in diarization_data["files"]:
            if audio_data.get("file") == audio_filename:
                segments = audio_data.get("turns", [])
                break
                
    meilleur_locuteur = "INCONNU"
    meilleur_chevauchement = 0
    
    for seg in segments:
        s_start = seg.get('start', 0)
        s_end = seg.get('end', 0)
        speaker = seg.get('speaker', 'INCONNU')
        
        # Formule de calcul du chevauchement entre l'entité et la prise de parole
        overlap_start = max(start_time, s_start)
        overlap_end = min(end_time, s_end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > meilleur_chevauchement:
            meilleur_chevauchement = overlap
            meilleur_locuteur = speaker
            
    return meilleur_locuteur

def map_chars_to_time(char_start, char_end, words_list):
    """
    Convertit la position d'un mot dans le texte (en index de caractères) 
    vers ses timestamps (début/fin) dans le fichier audio (en secondes).
    """
    start_time = float('inf')
    end_time = 0.0
    
    for w in words_list:
        # Vérifie si le mot de WhisperX se trouve dans la plage de caractères de l'entité
        if not (w['char_end'] <= char_start or w['char_start'] >= char_end):
            start_time = min(start_time, w['start'])
            end_time = max(end_time, w['end'])
            
    return start_time if start_time != float('inf') else 0.0, end_time

# BOUCLE PRINCIPALE ET TRAITEMENT
# ========================================
def process_single_file(fichier_whisper, diarization_data):
    """
    Fonction orchestratrice : prend un fichier WhisperX, extrait le texte, 
    cherche les entités, calcule les temps, trouve les locuteurs, et génère le JSON final.
    """
    print(f"\n==================================================")
    print(f"[TRAITEMENT] Analyse du fichier : {fichier_whisper}")
    print(f"==================================================")
    
    # Sécurisation du chemin : on isole le nom du fichier sans les dossiers parents
    nom_base = os.path.basename(fichier_whisper)
    
    # Extraction dynamique de l'identifiant pour le nom du fichier de sortie
    match_id = re.search(r'(\d{4})\.\d+\.json$', nom_base)
    identifiant = match_id.group(1) if match_id else nom_base.split('.')[0].replace('audio-', '')
    
    # 1. Pipeline NLP
    texte_complet, words_list = load_whisperx_and_build_offsets(fichier_whisper)
    mots_extraits = extract_metier_words_with_spacy(texte_complet)
    
    # Le fichier audio cherché dans la diarization est le même nom mais en .wav (sans le chemin)
    fichier_wav = nom_base.replace('.json', '.wav')
    
    resultats_finaux = []

    # 3. Consolidation des données pour chaque entité trouvée    
    for entite in mots_extraits:
        audio_start, audio_end = map_chars_to_time(entite['char_start'], entite['char_end'], words_list)
        locuteur = get_speaker_for_time(audio_start, audio_end, diarization_data, fichier_wav)
        
        resultats_finaux.append({
            'mot': entite['mot'],
            'categorie': entite['categorie'],
            'locuteur': locuteur,
            'debut_sec': round(audio_start, 2),  # Arrondi pour lisibilité
            'fin_sec': round(audio_end, 2),
            'source_detection': entite['source']
        })
        
    # Tri temporel pour avoir une lecture chronologique logique
    resultats_finaux.sort(key=lambda x: x['debut_sec'])
    
    # 4. Rendu visuel dans le terminal
    if resultats_finaux:
        print(f"\n  Liste des {len(resultats_finaux)} mots métiers extraits :")
        print("  " + "-" * 75)
        print(f"  | {'MOT TROUVÉ':<25} | {'SOURCE DE DÉTECTION':<25} | {'LOCUTEUR':<15} |")
        print("  " + "-" * 75)
        
        for res in resultats_finaux:
            # Tronquature pour l'affichage console si le mot est trop long
            mot_affiche = res['mot'][:22] + "..." if len(res['mot']) > 25 else res['mot']
            source_affiche = res['source_detection'][:22] + "..." if len(res['source_detection']) > 25 else res['source_detection']
            print(f"  | {mot_affiche:<25} | {source_affiche:<25} | {res['locuteur']:<15} |")
        
        print("  " + "-" * 75 + "\n")
    else:
        print("\n  [INFO] Aucun mot métier trouvé dans ce fichier.\n")
    
    # Sauvegarde JSON
    nom_sortie = f"mots_metiers_extraits_{identifiant}.json"
    with open(nom_sortie, 'w', encoding='utf-8') as f:
        json.dump(resultats_finaux, f, ensure_ascii=False, indent=4)
        
    print(f"  ✓ Export JSON réussi : {nom_sortie}\n")


def main():
    """
    Point d'entrée du programme. Charge la base de données Pyannote,
    scanne le dossier cible et propose un menu interactif à l'utilisateur.
    """
    fichier_diarization = "segmentation_audios_Nexsis.json"
    print(f"\n[INFO] Chargement de la base de diarization : {fichier_diarization}...")
    
    diarization_data = load_pyannote_diarization(fichier_diarization)
    if not diarization_data:
        return
        
    print("[INFO] Base de diarization chargée avec succès.")

    dossier_audio = "mes_audio"

    while True:
        # Vérifier si le dossier existe
        if not os.path.exists(dossier_audio):
            print(f"\n[ERREUR] Le dossier '{dossier_audio}' est introuvable. Veuillez le créer à la racine du script.")
            break

        # Lister uniquement les fichiers JSON dans le dossier
        fichiers_json = [f for f in os.listdir(dossier_audio) if f.endswith('.json')]
        fichiers_json.sort() # Trie par ordre alphabétique

        if not fichiers_json:
            print(f"\n[INFO] Aucun fichier .json n'a été trouvé dans le dossier '{dossier_audio}'.")
            break

        # Affichage du menu interactif
        print("\n" + "="*60)
        print(f" Fichiers disponibles dans '{dossier_audio}' :")
        print("="*60)
        for i, fichier in enumerate(fichiers_json, start=1):
            print(f" [{i}] {fichier}")
        print(" [0] Quitter le programme")
        print("="*60)

        entree = input("\nEntrez le numéro du fichier à analyser :\n> ").strip()
        
        # Gestion de la sortie
        if entree == '0' or entree.lower() in ['q', 'quit', 'quitter', 'exit']:
            print("Fin du programme. À bientôt !")
            break
            
        # Validation du choix
        try:
            choix = int(entree)
            if 1 <= choix <= len(fichiers_json):
                fichier_choisi = fichiers_json[choix - 1]
                # Reconstruire le chemin complet vers le fichier
                chemin_complet = os.path.join(dossier_audio, fichier_choisi)
                
                try:
                    process_single_file(chemin_complet, diarization_data)
                except Exception as e:
                    print(f"[ERREUR] Une erreur inattendue est survenue lors du traitement de {fichier_choisi} : {e}")
            else:
                print(f"[ERREUR] Veuillez entrer un numéro valide (entre 1 et {len(fichiers_json)}).")
        except ValueError:
            print("[ERREUR] Saisie invalide. Veuillez entrer un nombre.")

if __name__ == "__main__":
    main()