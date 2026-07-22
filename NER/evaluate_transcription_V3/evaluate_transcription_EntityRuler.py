import json
import re
import sys
import pympi
import unicodedata
import jiwer
import spacy
import argparse
import os

# nom du fichier : evaluate_transcription_EntityRuler.py
# commande : python evaluate_transcription_EntityRuler.py --eaf "Appel_3540_annotation_finale.eaf" --json "transcription_whisper_3540_brute.json"
# script basé NER (IA) avec Spacy et EntityRuler incorpore regex (metier)
# Configurer votre espace virtuel (audio_env)
# conda activate audio_env
# pip install jiwer
# pip install pympi-ling
# python -m pip install spacy
# python -m spacy download fr_core_news_sm
# Téléchargez le modèle d'Intelligence Artificielle en français :
# python -m spacy download fr_core_news_sm
# besoin d un fichier patterns_sdis.jsonl le Regex métier

# ==========================================
# 1. CHARGEMENT DE L'IA (spaCy + JSONL)
# ==========================================
print("[INIT] Chargement du modèle d'Intelligence Artificielle (spaCy)... patientez.")
nlp = spacy.load("fr_core_news_sm")

# Ajout du composant EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Chargement du fichier JSONL contenant toutes vos règles métiers
fichier_regles = "patterns_sdis.jsonl"
if os.path.exists(fichier_regles):
    ruler.from_disk(fichier_regles)
    print(f"[INIT] Règles métiers SDIS chargées depuis {fichier_regles}.")
else:
    print(f"[ATTENTION] Le fichier {fichier_regles} est introuvable. Seule l'IA statistique sera active.")


# ==========================================
# 2. GAZETTEER (Dictionnaire Géographique)
# ==========================================
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
        print(f"[INFO] {len(set(villes))} communes chargées depuis le fichier.")
    return list(set(villes))

def enlever_accents(texte):
    return ''.join(c for c in unicodedata.normalize('NFD', texte) if unicodedata.category(c) != 'Mn')

def mot_to_regex(mot):
    """Génère une Regex tolérante aux accents pour un seul mot."""
    base = enlever_accents(mot.lower().strip())
    mapping = {'e': '[eéèêë]', 'a': '[aàâä]', 'o': '[oôö]', 'i': '[iîï]', 'u': '[uùûü]', 'c': '[cç]'}
    return "".join(mapping.get(c, c) for c in base)

# Chargement des communes
GAZETTEER_COMMUNES_31 = charger_gazetteer("communes-haute-garonne.geojson.json")

# Injection dynamique des communes dans l'EntityRuler
patterns_communes_spacy = []
for commune in GAZETTEER_COMMUNES_31:
    morceaux = re.split(r'[\s\-]+', commune.strip())
    # On crée une liste de jetons Regex pour chaque morceau du nom de la commune
    pattern = [{"LOWER": {"REGEX": f"^{mot_to_regex(m)}$"}} for m in morceaux]
    patterns_communes_spacy.append({"label": "COMMUNES_31", "pattern": pattern})

ruler.add_patterns(patterns_communes_spacy)
print(f"[INFO] Communes intégrées avec succès à l'EntityRuler (Optimisation phonétique activée).")


# ==========================================
# 3. FONCTIONS DE BASE
# ==========================================
def normalize_text(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r"[^\w\s]", " ", text)
    disfluences = {"ah", "aie", "atchoum", "baf", "bah", "be", "ben", "bien", "bof", "bon ben", "bouh", "euh", "euf", "ha", "heu", "heueu", "he", "he bien", "hein", "hi", "ih", "hm", "hop", "hou", "hum", "hup", "la", "mah", "menfin", "mmm", "mouais", "moui", "of", "oh", "ok", "okay", "ouah", "ouais", "ouf", "ouille", "pff", "pouh", "snif", "tac", "toc", "wahou", "yeah", "zut", "zou"}
    for df in sorted(disfluences, key=len, reverse=True):
        pattern = r"\b" + re.escape(df) + r"\b"
        text = re.sub(pattern, " ", text)
    return " ".join(text.split())

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
        annotations.append({"start": debut_ms / 1000.0, "end": fin_ms / 1000.0, "text": texte.strip()})
    return annotations

# NOUVELLE FONCTION : Calcul du WER Global
def calculer_wer_global(elan_segments, whisper_words):
    """
    Calcule le Word Error Rate sur l'intégralité du texte, 
    sans distinction de locuteur, pour évaluer la transcription globale.
    """
    # 1. Reconstruire le texte de référence complet (trié chronologiquement)
    segments_tries = sorted(elan_segments, key=lambda x: x['start'])
    texte_reference = " ".join([seg['text'] for seg in segments_tries])
    ref_norm = normalize_text(texte_reference)
    
    # 2. Reconstruire le texte hypothèse complet (Whisper)
    texte_whisper = " ".join([w['word'] for w in whisper_words if 'word' in w])
    hyp_norm = normalize_text(texte_whisper)
    
    # 3. Calcul du WER global
    if not ref_norm: return 0
    return jiwer.wer(ref_norm, hyp_norm)


# ==========================================
# 4. MOTEUR HYBRIDE : ENTITY RULER + NER
# ==========================================
def est_faux_positif(mot_detecte):
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
    
    # Liste des labels de l'IA classique (pour les séparer de nos labels métiers)
    LABELS_IA = ["LOC", "ORG", "PER", "PERSON", "DATE", "TIME", "CARDINAL"]
    
    for segment in elan_segments:
        ref_texte = segment['text']
        
        infos_dans_ce_segment = []
        
        # --- PASSE UNIQUE : spaCy gère tout (Métier + IA) ---
        doc = nlp(ref_texte)
        
        for ent in doc.ents:
            texte_entite = ent.text.strip()
            # ---  Ignorer les "Divers" inutiles de spaCy ---
            if ent.label_ == "MISC":
                continue
            
            # CAS 1 : C'est un concept MÉTIFR (issu du JSONL ou des Communes)
            if ent.label_ not in LABELS_IA:
                if ent.label_ == "LOCALISATION_SPECIFIQUE" and est_faux_positif(texte_entite):
                    continue
                    
                # Nettoyage de l'apostrophe isolée en fin de mot
                mot_propre = re.sub(r"\s+(c|qu|l|d|s|m|t|n|j)$", "", texte_entite, flags=re.IGNORECASE).strip()
                infos_dans_ce_segment.append(("MÉTIER", ent.label_, mot_propre))
                
            # CAS 2 : C'est un concept IA Standard
            else:
                texte_sans_ponctuation = re.sub(r"[^\w\s]", "", texte_entite)
                if len(texte_sans_ponctuation) <= 2:
                    continue
                
                tokens = re.findall(r'\b\w+\b', texte_entite.lower())
                mots_non_mesure = {"sur", "sous", "dans", "vers", "près", "de", "du", "la", "le", "les", "lès"}
                mots_mesure_pure = {"mètre", "metre", "mètres", "metres", "km", "kilomètre", "kilometre", "kilomètres", "kilometres"}
                nombres_lettres = {"un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "vingt", "trente", "quarante", "cinquante", "soixante", "cent", "mille"}
                
                tokens_contenu = [t for t in tokens if t not in mots_non_mesure]
                if tokens_contenu and all(t in nombres_lettres or t in mots_mesure_pure or t.isdigit() for t in tokens_contenu):
                    continue
                
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
                
                infos_dans_ce_segment.append(("NER", categorie_ner, texte_entite))
            
        if not infos_dans_ce_segment:
            continue
            
        # --- RÉCUPÉRATION DES MOTS DE LA MACHINE (Avec la marge) ---
        mots_whisper_fenetre = [w['word'].lower() for w in whisper_words if 'start' in w and 'end' in w and w['start'] < (segment['end'] + marge_secondes) and w['end'] > (segment['start'] - marge_secondes)]
        texte_whisper_fenetre = " ".join(mots_whisper_fenetre)
        
        # --- VÉRIFICATION ---
        for type_detect, categorie, mot_exact in infos_dans_ce_segment:
            infos_attendues += 1
            info_sauvee = False
            
            # Simple vérification par inclusion
            if mot_exact.lower() in texte_whisper_fenetre:
                info_sauvee = True
                    
            if info_sauvee:
                infos_trouvees += 1
                mots_sauves.append((categorie, mot_exact))
                print(f"✅ [{type_detect}] {categorie} SAUVÉ ! Humain: '{ref_texte}' -> Whisper a capté '{mot_exact}'.")
            else:
                mots_perdus.append((categorie, mot_exact))
                print(f"❌ [{type_detect}] {categorie} PERDU ! Humain: '{ref_texte}' -> Whisper a raté '{mot_exact}'.")

    score = (infos_trouvees / infos_attendues) * 100 if infos_attendues > 0 else 0
    return score, mots_sauves, mots_perdus


# ==========================================
# 5. L'ORCHESTRATEUR (MAIN)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation Hybride : ELAN vs WhisperX via EntityRuler")
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
        
        # --- NOUVEAU : Calcul du WER Global ---
        elan_complet = elan_operateur + elan_requerant
        wer_global = calculer_wer_global(elan_complet, whisper_data)
        
        print("\n===========================================")
        print(" BILAN GLOBAL COMPARATIF")
        print("===========================================")
        print(f"WER Global (Texte complet) : {wer_global:.2f}")
        print(f"Score Hybride Opérateur    : {score_op:.1f}%")
        print(f"Score Hybride Requérant    : {score_req:.1f}%")

        print("\n===========================================")
        print(" SYNTHÈSE DES INFORMATIONS MÉTIER (SDIS 31)")
        print("===========================================")
        
        tous_sauves = sauves_op + sauves_req
        tous_perdus = perdus_op + perdus_req
        
        print(f"🟢 INFORMATIONS SÉCURISÉES PAR L'IA ({len(tous_sauves)}) :")
        if tous_sauves:
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
