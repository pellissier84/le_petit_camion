import json
import re
import sys
import pympi
import unicodedata
import jiwer
import spacy
import argparse
import os


# nom du fichier : python evaluate_transcription_NER_REGEX.py
# commande : python evaluate_transcription_NER_REGEX_V1.py --eaf "Appel_3540_annotation_finale.eaf" --json "transcription_whisper_3540_brute.json"
# script  hybride regex (metier) et NER (IA)
# Configurer votre espace virtuel (audio_env)
# conda activate audio_env
# pip install jiwer
# pip install pympi-ling
# python -m pip install spacy
# python -m spacy download fr_core_news_sm
# Téléchargez le modèle d'Intelligence Artificielle en français :
# python -m spacy download fr_core_news_sm

# 1. Chargement de l'IA (spaCy)
print("Chargement du modèle d'Intelligence Artificielle (spaCy)... patientez.")
# Chargement du cerveau NER français
nlp = spacy.load("fr_core_news_sm")

# 2. Chargement automatique du Gazetteer (Communes 31)
# ==========================================
# 2 GAZETTEER (Dictionnaire Géographique)
# ==========================================
def charger_gazetteer(chemin_fichier):
    villes = []
    if os.path.exists(chemin_fichier):
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            # Si JSON
            if chemin_fichier.endswith('.json'):
                donnees = json.load(f)
                for feature in donnees.get('features', []):
                    ville = feature.get('properties', {}).get('nom_commune', '')
                    if ville:
                        villes.append(ville.strip().lower().replace("-", " "))
            # Si CSV (optionnel)
            elif chemin_fichier.endswith('.csv'):
                import csv
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    ville = row.get('nom_commune', '')
                    if ville:
                        villes.append(ville.strip().lower().replace("-", " "))
        print(f"[INFO] {len(set(villes))} communes chargées depuis le fichier.")
    return list(set(villes))

# Chargement initial des communes depuis votre fichier GeoJSON
GAZETTEER_COMMUNES_31 = charger_gazetteer("communes-haute-garonne.geojson.json")

def enlever_accents(texte):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texte)
        if unicodedata.category(c) != 'Mn'
    )

#  CONVERSION DE CHAQUE COMMUNE EN REGEX DYNAMIQUE
def commune_to_regex(nom_commune):
    r"""
    Transforme 'lagardelle sur lèze' en 'l[aàâä]g[aàâä]rd[eéèêë]ll[eéèêë][\s-]+s[uùûü]r[\s-]+l[eéèêë]z[eéèêë]'
    pour tolérer absolument toutes les variations d'accents, d'espaces ou de tirets de l'oral.
    """
    # 1. On repart d'une base propre en minuscules et sans aucun accent historique
    base = enlever_accents(nom_commune.lower().strip())
    
    # 2. CORRECTION PYTHON 3.14 : On double-échappe le \s pour la chaîne de remplacement
    # Le tiret mis à la fin [-] n'a plus besoin d'être échappé, ce qui évite le bug du parseur.
    base = re.sub(r'[\s\-]+', r'[\\s-]+', base)
    
    # 3. On mappe chaque voyelle vers sa classe de caractères d'accents
    mapping = {
        'e': '[eéèêë]',
        'a': '[aàâä]',
        'o': '[oôö]',
        'i': '[iîï]',
        'u': '[uùûü]',
        'c': '[cç]'
    }
    
    return "".join(mapping.get(c, c) for c in base)

# Transformation de votre liste brute en une liste de patterns de Regex optimisés
patterns_communes = list(set(commune_to_regex(c) for c in GAZETTEER_COMMUNES_31))
print(f"[INFO] {len(patterns_communes)} patterns uniques générés pour la Haute-Garonne.")

#  RECONSTRUCTION DE LA REGEX GLOBALE INTÉGRALE
REGEX_COMMUNES = r"\b(" + "|".join(patterns_communes) + r")\b"
print(f"[INFO] Regex des communes construite avec succès (Optimisation phonétique activée).")


# 3. Ontologie Métier (Regex)
# ==========================================
# 3 ONTOLOGIE MÉTIER (SDIS 31) - REGEX
# ==========================================
CONCEPTS_URGENCE = {
    "CONSCIENCE_NEURO": [
        # --- Formes négatives (Sécurisées avec \s+) ---
        r"ne\s+va\s+pas\s+bien", r"ne\s+se\s+sent\s+pas\s+bien", r"ne\s+réagit\s+pas", 
        r"ne\s+bouge\s+(plus|pas)",
        # Gestion de : "ne répond pas", "ne me répond pas", "ne repondent pas", "ne m'a pas repondu" etc.
        r"ne\s+(me\s+)?r[eé]pond(s|ent|ait)?\s+pas",
        
        # --- États du corps ---
        r"allongée?\s+au\s+sol", r"tombée?", r"inerte", 
        r"perte\s+de\s+(connaissance|conscience)", 
        r"tremble[s]?", r"tremblement[s]?", r"crise\s+de\s+(tremblements?|spasmophilie|tétanie)", 
        r"crise\s+d[’']épilepsie", r"malaise", r"paralys(é|ée|ie)", r"yeux\s+fixes",
        
        # --- Nouveaux termes  ---
        r"troubles?\s+sensoriels?", r"convulsions?", r"troubles?\s+d[’']élocution",
        r"pupilles\s+dilatées?", r"vision\s+troublée?", r"troubles?\s+de\s+la\s+vision",
        r"perte\s+d[’']équilibre", r"vomissements?", r"vertiges?", r"mal\s+de\s+tête", r"maux\s+de\s+tête",
        r"sous\s+emprise\s+de\s+(drogues?|alcool|stupéfiants?|médicaments?)", r"sous\s+(traitement|médicaments?)",
        r"électrocutée?", r"électrocution", r"(ingestion|inhalation|injection)\s+de\s+produits\s+toxiques",
        r"évanouissement", r"évanouie?", r"sueurs?", r"pâleur|paleur", r"blancheur|blanc|blanche", r"livide"
    ],
    "DETRESSE_VITALE": [
        r"flaque[s]? de sang", r"baigne dans son sang", r"se vide de son sang",
        r"saignement[s]? (des oreilles|par la bouche|abondant)?", 
        r"sang", r"coupure", r"entaille", r"taillader", r"objet dans la plaie",
        r"difficulté[s]? [àa] respirer", r"ne respire pas", r"asphyxie", r"toux",
        r"son cœur ne bat pas",
        # --- Nouveaux termes ---
	r"anémi(é|ie)",r"hémorragie", r"métroragie[s]",
        r"œdème|oedeme", r"malade du (cœur|coeur)", r"malade cardiaque", r"diabète|diabétique",
	# --- MALADIES NEURODÉGÉNÉRATIVES ---
        # Capte : "maladie d'alzheimer", "maladie de lewy", "maladie de parkinson", "maladie d alzheimer" 
        r"maladie\s+d[e'’\s]+(alzheimer|lewy|parkinson|charcot)",
        
        # Capte les gens qui disent directement le nom : "il a alzheimer", "elle a un parkinson"
        r"\b(alzheimer|parkinson)\b",
        
        # Capte spécifiquement "corps de lewy"
        r"corps\s+de\s+lewy",
        r"douleur[s]? dans la poitrine", r"respire (avec difficulté|avec bruit|rapidement)", r"s[’']étouffe",
        r"saignement (en jet|en saccade[s]?|sous pression)", r"brûlure[s]?|brulure[s]?", r"plaie profonde",
        r"j[’']ai entrepris une r[eé]animation", r"j[’']ai posé un dae"
    ],
    "TRAUMATISME": [
        r"chute( de sa hauteur| des escaliers| d[’']une échelle| sur la tête)?",
        r"déformation( du visage| de la bouche)?", r"douleur( au dos)?",
        r"cassé", r"déplacé", r"amputé", r"coupé",
        # --- Nouveaux termes ---
        r"fortes douleurs?|douleurs? vives?", r"incapable (de|à) bouger", r"incapacité [àa] bouger",
        r"amputation", r"luxation", r"déboîte(r|ment)? l[’']articulation"
    ],
    "INCENDIE_GAZ": [
        r"fumée( opaque| noire| grise| blanche| irrespirable)?", r"fumée qui pique les yeux",
        r"odeur de (gaz|brûlé|nauséabonde)", r"intoxication", r"feu", r"flamme[s]?", 
        r"explosion", r"gaz (toxique|irritant)"
    ],
    "ACCIDENT_ROUTE": [
        r"accident( routier| grave| simple| léger)?", r"carambolage", r"tonneaux",
        r"éjecté", r"coincé", r"personne (éjectée|coincée|sortie)",
	r"(grande|petite) vitesse", r"cinétique importante",
        r"poids lourd", r"trottinette( électrique)?", r"voiture[s]?", r"camion( petit| gros)?", 
        r"bus", r"tracteur", r"scooter", r"vélo", r"cycliste", r"piéton"
    ],
    "LOCALISATION_SPECIFIQUE": [
        r"en ville", r"au rond[-]point", r"au croisement",
        r"autoroute", r"voie( rapide)?", r"pont", r"péage",
        
        # --- NOUVEAUX TERMES : REZ-DE-CHAUSSÉE ---
        r"rez\s*de\s*chauss[eé]e", 
        r"\brdc\b", 
        r"niveau\s+(zero|0)",
        
        # --- NOUVEAUX TERMES : ÉTAGES ET APPARTEMENTS ---
        r"\b[àa]\s+l['’]?étage\b",
        r"\b(\d{1,2}(er|ème|eme|e)?|(premier|deuxième|troisième|quatrième|cinquième|sixième|septième|huitième|neuvième|dixième|onzième|douzième)|(un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze))\s+étage\b",
        r"\bétage\s+(\d{1,2}|un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\b",
        r"\bappart(ement)?\s+(numéro\s+|n[°o]\s+)?(\d{1,4}|[a-z])\b",

        # --- ROUTES SPÉCIFIQUES ---
        r"d\s*622", r"d\s*six\s*cent\s*vingt\s*deux",
        r"\b(autoroute|départementale|nationale|rn|rd)\s+(un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|vingt|trente|quarante|cinquante|soixante|cent|mille|\d+)\b",
        r"\b(a|d|n)\s*\d{1,4}\b",
        
        # --- NOUVEAU : PATTERNS D'ADRESSES RÉSIDENTIELLES ---
        r"\b(résidence|chemin|rue|avenue|impasse|place|batiment|boulevard|allée)\s+(?:de\s+|du\s+|la\s+|des\s+)?[\w\s]{2,20}\b",
        
        # --- Nouveaux termes (Contexte & Géographie) ---
        r"dénivelé important[e]?",
        r"en contre[-]bas", r"dans un fossé", r"fossé profond", r"cours? d[’']eau",
        r"\b(torrent|rivière|lac|fleuve|trou|puits)\b",
        r"rue étroite", r"chemin de terre", r"impasse", r"tunnel",r"batiment",r"maison"
    ],
    "COMMUNES_31": [
        REGEX_COMMUNES
    ],
    "ARMES_VIOLENCE": [
        r"arme à feu", r"fusil", r"pistolet", r"kalashnikov"
    ],
    "IMPLIQUES": [
        r"enfant", r"bébé", r"adulte", r"père", r"mère", r"fils", r"fille", 
        r"voisin", r"personne", r"patient[e]?",
        # --- Nouveaux termes ---
        r"nourrisson[s]?", r"grand[-]père", r"grand[-]mère", r"personne[s]? âgée[s]?",
        r"personne handicapée", r"avec un handicap", r"en fauteuil roulant",
        r"malade[s]?", r"piéton|pieton", r"marcheur[s]?", r"randonneur[s]?", r"cycliste[s]?", r"motard[s]?"
    ],
    "AGE_VICTIME": [
        # 1. Le cas  : Whisper a convertit en chiffres (ex: "70 ans", "8 mois", "1 an")
        r"\b\d{1,3}\s+(ans?|mois|semaines?|jours?)\b",
        
        # 2. Le filet de sécurité : Whisper a laissé en lettres (ex: "soixante dix ans", "trois mois")
        # Le [\s\-]? permet de capter "soixante dix", "soixante-dix" ou "soixantedix"
        r"\b(un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|vingt|trente|quarante|cinquante|soixante|soixante[\s\-]?dix|quatre[\s\-]?vingt[s]?|quatre[\s\-]?vingt[\s\-]?dix|cent)\s+(ans?|mois)\b"
    ],
    "MESURES": [
        # 1. Capte tous les chiffres + unités (ex: "25 mètres", "1.5 km", "40 cm", "100m")
        # Explication : \d+ (chiffres) + optionnellement une virgule et des chiffres + l'unité (avec ou sans 's')
        r"\b\d+(?:[.,]\d+)?\s*(?:mètres?|metres?|centimètres?|centimetres?|kilomètres?|kilometres?|km|cm|m)\b",
        
        # 2. Capte les nombres dictés à l'oral + unités (ex: "vingt cinq mètres", "trois kilomètres", "une centaine de metres")
        # Explication : Un mot de nombre + un espace possible de 0 à 20 caractères (pour capter le "cinq" de "vingt-cinq") + l'unité
        r"\b(?:un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|vingt|trente|quarante|cinquante|soixante|cent|mille|dizaine|centaine)[\w\s-]{0,20}(?:mètres?|metres?|centimètres?|centimetres?|kilomètres?|kilometres?|km|cm)\b"
    ],
    "DEMANDE_MOYENS": [
        # --- Nouveaux termes de la section Divers ---
        r"demande d[’']ambulance", r"médecin|docteur", r"infirmi[eèé]re?s?", r"demande de secours",
        r"camion de feu", r"camion pour combattre l[’']incendie", r"lutter contre l[’']incendie",
        r"(gros|petit) camion", r"grande échelle", r"voiture de pompier[s]?", r"véhicule de secours"
    ]
}

# 4. Vos fonctions de traitement (normalize_text, load_whisperx, etc.)
# ==========================================
# 4. FONCTIONS DE BASE
# ==========================================
def normalize_text(text):
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r"[^\w\s]", " ", text)
    disfluences = {
        "ah", "aie", "atchoum", "baf", "bah", "be", "ben", "bien", "bof",
        "bon ben", "bouh", "euh", "euf", "ha", "heu", "heueu", "he", "he bien",
        "hein", "hi", "ih", "hm", "hop", "hou", "hum", "hup", "la", "mah", "menfin", "mmm", "mouais",
        "moui", "of", "oh", "ok", "okay", "ouah", "ouais", "ouf", "ouille",
        "pff", "pouh", "snif", "tac", "toc", "wahou", "yeah", "zut", "zou"
    }
    for df in sorted(disfluences, key=len, reverse=True):
        pattern = r"\b" + re.escape(df) + r"\b"
        text = re.sub(pattern, " ", text)
    mots = text.split()
    return " ".join(mots)

def load_whisperx(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tous_les_mots = []
    for segment in data.get('segments', []):
        if 'words' in segment:
            tous_les_mots.extend(segment['words'])
    return tous_les_mots

def load_elan_data(filepath, tier_name):
    eaf = pympi.Elan.Eaf(filepath)
    if tier_name not in eaf.get_tier_names():
        return []
    annotations = []
    for debut_ms, fin_ms, texte in eaf.get_annotation_data_for_tier(tier_name):
        annotations.append({
            "start": debut_ms / 1000.0,
            "end": fin_ms / 1000.0,
            "text": texte.strip()
        })
    return annotations

def align_and_evaluate(elan_segments, whisper_words):
    total_wer = 0
    segments_valides = 0
    for segment in elan_segments:
        ref_norm = normalize_text(segment['text'])
        if not ref_norm: continue
        matching_words = [w['word'] for w in whisper_words if 'start' in w and 'end' in w and w['start'] < segment['end'] and w['end'] > segment['start']]
        hyp_norm = normalize_text(" ".join(matching_words)) or " "
        total_wer += jiwer.wer(ref_norm, hyp_norm)
        segments_valides += 1
    return total_wer / segments_valides if segments_valides > 0 else 0

# 5. Fonction hybride (avec extraction du mot exact et filtre NER robuste)
# ==========================================
# 5. MOTEUR HYBRIDE : REGEX + NER (spaCy)
# ==========================================
def est_faux_positif(mot_detecte):
    """
    Filtre métier pour rejeter les morceaux de phrases capturés par erreur 
    à cause de la grammaire française de l'oral.
    """
    mots_interdits = ["tant que", "tant qu", "pour", "c'est", "c est", "donc", "a un", "le", "la"]
    for interdit in mots_interdits:
        if interdit in mot_detecte.lower():
            return True
    return False

def evaluate_hybride(elan_segments, whisper_words, marge_secondes=3.0):
    infos_attendues = 0
    infos_trouvees = 0
    mots_sauves = []
    mots_perdus = []
    
    for segment in elan_segments:
        ref_texte = segment['text']
        ref_texte_lower = ref_texte.lower()
        
        infos_dans_ce_segment = []
        
        # --- PASSE 1 : Détection par Regex (Ontologie SDIS) ---
        for categorie, patterns in CONCEPTS_URGENCE.items():
            for pattern in patterns:
                match = re.search(pattern, ref_texte_lower)
                if match:
                    # 1. Capture du mot exact et suppression des espaces inutiles aux extrémités
                    mot_exact_trouve = match.group(0).strip() 
                    
                    # 2. FILTRE ANTI-BRUIT : Rejette les "place tant qu'" ou "a un"
                    if categorie == "LOCALISATION_SPECIFIQUE" and est_faux_positif(mot_exact_trouve):
                        continue
                        
                    # 3. NETTOYAGE DES APOSTROPHES : Coupe la lettre isolée à la fin (ex: "eris c" devient "eris")
                    mot_exact_trouve = re.sub(r"\s+(c|qu|l|d|s|m|t|n|j)$", "", mot_exact_trouve).strip()
                    
                    # On sauvegarde (Type, Catégorie, Le Pattern Regex, Le Vrai Mot)
                    infos_dans_ce_segment.append(("REGEX", categorie, pattern, mot_exact_trouve))
                    
        # --- PASSE 2 : Détection par NER (spaCy) ---
        doc = nlp(ref_texte)
        for ent in doc.ents:
            # On élargit les étiquettes ciblées par l'IA
            if ent.label_ in ["LOC", "ORG", "PER", "PERSON", "DATE", "TIME", "CARDINAL"]:
                texte_entite = ent.text.lower().strip()
                
                # Filtre 1 : Ignorer les micro-mots (bruit de l'oral comme "d", "ca", "qu'")
                texte_sans_ponctuation = re.sub(r"[^\w\s]", "", texte_entite)
                if len(texte_sans_ponctuation) <= 2:
                    continue
                
                # Filtre 2 (Conservé de votre version originale) : Ignorer les mesures pures
                tokens = re.findall(r'\b\w+\b', texte_entite)
                mots_non_mesure = {"sur", "sous", "dans", "vers", "près", "de", "du", "la", "le", "les", "lès"}
                mots_mesure_pure = {"mètre", "metre", "mètres", "metres", "km", "kilomètre", "kilometre", "kilomètres", "kilometres"}
                nombres_lettres = {"un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "vingt", "trente", "quarante", "cinquante", "soixante", "cent", "mille"}
                
                tokens_contenu = [t for t in tokens if t not in mots_non_mesure]
                
                # Si l'entité ne contient QUE des nombres/mesures, on l'ignore
                if tokens_contenu and all(t in nombres_lettres or t in mots_mesure_pure or t.isdigit() for t in tokens_contenu):
                    continue
                
                # --- TRI POUR LE RAPPORT MÉTIER ---
                if ent.label_ in ["LOC", "ORG"]:
                    categorie_ner = f"LIEU_INCONNU_IA ({ent.label_})"
                elif ent.label_ in ["PER", "PERSON"]:
                    categorie_ner = f"PERSONNE_IA ({ent.label_})"
                elif ent.label_ in ["DATE", "TIME"]:
                    categorie_ner = f"TEMPORALITE_IA ({ent.label_})"
                elif ent.label_ == "CARDINAL":
                    categorie_ner = f"QUANTITE_IA ({ent.label_})"
                else:
                    categorie_ner = f"AUTRE_IA ({ent.label_})"
                
                # On sauvegarde l'entité avec sa nouvelle catégorie lisible
                infos_dans_ce_segment.append(("NER", categorie_ner, texte_entite, texte_entite))
            
        if not infos_dans_ce_segment:
            continue
            
        # --- RÉCUPÉRATION DES MOTS DE LA MACHINE (Avec la marge) ---
        mots_whisper_fenetre = [w['word'].lower() for w in whisper_words if 'start' in w and 'end' in w and w['start'] < (segment['end'] + marge_secondes) and w['end'] > (segment['start'] - marge_secondes)]
        texte_whisper_fenetre = " ".join(mots_whisper_fenetre)
        
        # --- VÉRIFICATION ---
        for type_detect, categorie, recherche, mot_exact in infos_dans_ce_segment:
            infos_attendues += 1
            info_sauvee = False
            
            if type_detect == "REGEX":
                if re.search(recherche, texte_whisper_fenetre):
                    info_sauvee = True
            elif type_detect == "NER":
                if recherche in texte_whisper_fenetre:
                    info_sauvee = True
                    
            if info_sauvee:
                infos_trouvees += 1
                mots_sauves.append((categorie, mot_exact))
                print(f"✅ [{type_detect}] {categorie} SAUVÉ ! Humain: '{ref_texte}' -> Machine a capté '{mot_exact}'.")
            else:
                mots_perdus.append((categorie, mot_exact))
                print(f"❌ [{type_detect}] {categorie} PERDU ! Humain: '{ref_texte}' -> Machine a raté '{mot_exact}'.")

    score = (infos_trouvees / infos_attendues) * 100 if infos_attendues > 0 else 0
    return score, mots_sauves, mots_perdus

# 6. Main
# ==========================================
# 6. L'ORCHESTRATEUR (MAIN)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation Hybride : ELAN vs WhisperX")
    parser.add_argument("--eaf", required=True, help="Le chemin exact vers le fichier ELAN (.eaf)")
    parser.add_argument("--json", required=True, help="Le chemin exact vers le fichier WhisperX (.json)")
    
    args = parser.parse_args()
    fichier_eaf = args.eaf
    fichier_json = args.json
    
    try:
        print(f"\n[CHARGEMENT] Lecture de {fichier_eaf} et {fichier_json}...")
        whisper_data = load_whisperx(fichier_json)
        elan_operateur = load_elan_data(fichier_eaf, tier_name="OP-SDIS1")
        elan_requerant = load_elan_data(fichier_eaf, tier_name="Requerant1")
        
        print(f"\n===========================================")
        print(f" ANALYSE HYBRIDE (MÉTIER + IA)")
        print(f"===========================================")
        
        print("\n--- OPÉRATEUR ---")
        score_op, sauves_op, perdus_op = evaluate_hybride(elan_operateur, whisper_data)
        
        print("\n--- REQUÉRANT ---")
        score_req, sauves_req, perdus_req = evaluate_hybride(elan_requerant, whisper_data)
        
        wer_op = align_and_evaluate(elan_operateur, whisper_data)
        wer_req = align_and_evaluate(elan_requerant, whisper_data)
        
        print("\n===========================================")
        print(" BILAN GLOBAL COMPARATIF")
        print("===========================================")
        print(f"Opérateur -> WER (Erreurs) : {wer_op:.2f} | Score Hybride : {score_op:.1f}%")
        print(f"Requérant -> WER (Erreurs) : {wer_req:.2f} | Score Hybride : {score_req:.1f}%")

        # --- NOUVEAU : LE RAPPORT D'EXTRACTION MÉTIER ---
        print("\n===========================================")
        print(" SYNTHÈSE DES INFORMATIONS MÉTIER (SDIS 31)")
        print("===========================================")
        
        tous_sauves = sauves_op + sauves_req
        tous_perdus = perdus_op + perdus_req
        
        print(f"🟢 INFORMATIONS SÉCURISÉES PAR L'IA ({len(tous_sauves)}) :")
        if tous_sauves:
            # On retire les doublons et on trie par ordre alphabétique
            for categorie, mot in sorted(list(set(tous_sauves))):
                print(f"   - {categorie:30} : '{mot}'")
        else:
            print("   (Aucune information métier détectée)")

        print(f"\n🔴 INFORMATIONS MANQUANTES (ERREURS WHISPER) ({len(tous_perdus)}) :")
        if tous_perdus:
            for categorie, mot in sorted(list(set(tous_perdus))):
                print(f"   - {categorie:30} : '{mot}'")
        else:
            print("   (Aucune perte d'information !)")
        print("\n")
        
    except FileNotFoundError as e:
        print(f"\nERREUR CRITIQUE : Impossible de trouver l'un des fichiers.")
        print(f"Détail technique : {e}")
        print("Vérifiez l'orthographe de vos fichiers dans la commande.")
