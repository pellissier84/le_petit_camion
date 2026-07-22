import json
import streamlit as st
import matplotlib.pyplot as plt


# visualisation des wav diariser par pyannote
# nom du fichier : segmentation_audios_Nexsis.json
# installation nécessaire : pip install streamlit matplotlib
# utilisation du script app.py dans le meme dossier que le json
# depuis un terminal : streamlit run app.py
# Une page web va s'ouvrir automatiquement dans votre navigateur. 
# Sur la gauche, vous aurez un menu déroulant listant vos differents fichiers. 
# Dès que vous en sélectionnez un, le chronogramme et les temps de parole s'affichent instantanément, 
# proprement, et sans inonder votre ordinateur de centaines d'images.


# Configuration de la page
st.set_page_config(page_title="Explorateur Diarisation", layout="wide")
st.title("🎙️ Explorateur de Segmentation Audio")

# 1. Charger les données en mémoire (avec cache pour la rapidité)
@st.cache_data
def charger_donnees(chemin):
    with open(chemin, 'r', encoding='utf-8') as f:
        return json.load(f)

# Remplacez par le bon nom de fichier si besoin
try:
    data = charger_donnees('segmentation_audios_Nexsis.json')
except FileNotFoundError:
    st.error("Fichier JSON introuvable. Vérifiez le nom et l'emplacement.")
    st.stop()

# 2. Créer un menu déroulant pour choisir le fichier
noms_fichiers = [f['file'] for f in data.get('files', [])]
st.sidebar.header("Navigation")
fichier_selectionne = st.sidebar.selectbox(f"Choisissez un fichier ({len(noms_fichiers)} au total) :", noms_fichiers)

# 3. Récupérer les données du fichier sélectionné
fichier_cible = next(f for f in data['files'] if f['file'] == fichier_selectionne)
segments = fichier_cible.get('turns', [])
locuteurs = sorted(list(set([seg['speaker'] for seg in segments])))

st.subheader(f"Analyse en cours : {fichier_selectionne}")

if not segments:
    st.warning("Aucun segment de parole trouvé pour ce fichier.")
else:
    # 4. Générer le graphique uniquement pour CE fichier
    hauteur = max(2.0, len(locuteurs) * 0.8)
    fig, ax = plt.subplots(figsize=(12, hauteur))
    
    couleurs = plt.cm.get_cmap('tab10', len(locuteurs))
    dico_couleurs = {loc: couleurs(i) for i, loc in enumerate(locuteurs)}
    
    for seg in segments:
        debut, fin, locuteur = seg['start'], seg['end'], seg['speaker']
        y_pos = locuteurs.index(locuteur)
        ax.broken_barh([(debut, fin - debut)], (y_pos - 0.4, 0.8), 
                       facecolors=dico_couleurs[locuteur], edgecolor='black', linewidth=0.5)
        
    ax.set_yticks(range(len(locuteurs)))
    ax.set_yticklabels(locuteurs)
    ax.set_xlabel('Temps (secondes)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 5. Afficher le graphique dans l'application
    st.pyplot(fig)
    
    # Bonus : Afficher un résumé textuel des temps de parole
    st.markdown("### 📊 Répartition du temps de parole")
    temps_parole = {loc: 0 for loc in locuteurs}
    for seg in segments:
        temps_parole[seg['speaker']] += (seg['end'] - seg['start'])
        
    for loc, temps in temps_parole.items():
        st.write(f"- **{loc}** : {temps:.1f} secondes")
