# app.py
# ------------------------------------------------------------
# Application Streamlit pour extraire le texte de PDF et l'exporter en .txt
# PyMuPDF (fitz) uniquement
# - Gestion des métadonnées détectées (affichage, sélection et injection dans la sortie texte)
# - Heuristique simple pour les PDF en colonnes (1 ou 2 colonnes) avec PyMuPDF (blocs)
# - Export individuel ou en archive ZIP
# - Avertissement si le document semble être un scan sans texte (pas d'OCR ici)
# - Bandeau sous le titre avec www.codeandcortex.fr et le profil LinkedIn (logo + lien)
# ------------------------------------------------------------

import io
from pathlib import Path
from datetime import datetime
import zipfile

import streamlit as st

# ------------------------------------------------------------
# Dépendance obligatoire : PyMuPDF
# ------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# ------------------------------------------------------------
# Fonctions utilitaires (toutes en français)
# ------------------------------------------------------------

def lire_metadonnees_pdf(pdf_bytes: bytes) -> dict:
    """Lire les métadonnées du PDF avec PyMuPDF et retourner un dictionnaire plat."""
    meta = {
        "Titre": "",
        "Auteur": "",
        "Sujet": "",
        "Mots-clés": "",
        "Créateur": "",
        "Producteur": "",
        "Créé le": "",
        "Modifié le": "",
    }
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        md = doc.metadata or {}
        # Clés PyMuPDF usuelles
        correspondance = {
            "title": "Titre",
            "author": "Auteur",
            "subject": "Sujet",
            "keywords": "Mots-clés",
            "creator": "Créateur",
            "producer": "Producteur",
            "creationDate": "Créé le",
            "modDate": "Modifié le",
        }
        for k_src, k_dst in correspondance.items():
            if k_src in md and md[k_src]:
                meta[k_dst] = str(md[k_src])
    return meta


def detecter_si_pdf_scanné(pdf_bytes: bytes, pages_test: int = 3, seuil_caractères: int = 40) -> bool:
    """Déterminer rapidement si le PDF contient très peu de texte extractible (probablement un scan)."""
    total = 0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        nb = min(len(doc), pages_test)
        for i in range(nb):
            t = doc[i].get_text("text") or ""
            total += len(t.strip())
    return total < seuil_caractères


def extraire_texte_pymupdf(pdf_bytes: bytes, mode_colonnes: str = "auto") -> str:
    """Extraire le texte via PyMuPDF avec une heuristique de colonnes ('auto' | '1' | '2')."""
    out = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            try:
                blocs_raw = page.get_text("blocks")  # liste de tuples
            except Exception:
                out.append(page.get_text("text") or "")
                out.append("\n\n")
                continue

            # Conversion des blocs en dicts lisibles
            blocs = []
            for b in blocs_raw:
                if len(b) >= 5 and isinstance(b[4], str):
                    blocs.append(
                        {"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "text": (b[4] or "").strip()}
                    )

            # Filtrage des blocs vides
            blocs = [b for b in blocs if b["text"]]

            if not blocs:
                out.append(page.get_text("text") or "")
                out.append("\n\n")
                continue

            largeur = page.rect.width
            milieu = largeur / 2.0

            # Choix du mode colonnes
            mode = mode_colonnes
            if mode == "auto":
                gauche = sum(1 for b in blocs if b["x1"] <= milieu * 0.98)
                droite = sum(1 for b in blocs if b["x0"] >= milieu * 1.02)
                total_blocs = len(blocs)
                mode = "2" if (total_blocs >= 6 and gauche >= 2 and droite >= 2) else "1"

            if mode == "2":
                blocs_g = [b for b in blocs if b["x1"] <= milieu]
                blocs_d = [b for b in blocs if b["x0"] > milieu]
                blocs_g.sort(key=lambda b: (round(b["y0"], 1), round(b["x0"], 1)))
                blocs_d.sort(key=lambda b: (round(b["y0"], 1), round(b["x0"], 1)))
                texte_page = "\n".join(b["text"] for b in blocs_g) + "\n\n" + "\n".join(b["text"] for b in blocs_d)
            else:
                blocs.sort(key=lambda b: (round(b["y0"], 1), round(b["x0"], 1)))
                texte_page = "\n".join(b["text"] for b in blocs)

            out.append(texte_page)
            out.append("\n\n")
    return "".join(out)


def formater_sortie_texte(nom_fichier: str, texte: str, inclure_meta: bool, meta: dict, champs_meta_selectionnes: list) -> str:
    """Assembler le texte final avec ou sans métadonnées en tête."""
    header = []
    if inclure_meta:
        header.append(f"Fichier source : {nom_fichier}")
        for champ in champs_meta_selectionnes:
            if champ in meta and str(meta[champ]).strip():
                header.append(f"{champ} : {meta[champ]}")
        header.append(f"Date d'extraction : {datetime.now().isoformat(timespec='seconds')}")
        header.append("-" * 60)
    corps = texte or ""
    if header:
        return "\n".join(header) + "\n\n" + corps
    return corps


# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------

st.set_page_config(page_title="Extraction PDF → Texte", page_icon="📄", layout="wide")

if fitz is None:
    st.error("PyMuPDF (fitz) est requis mais indisponible. Ajoutez 'PyMuPDF' dans requirements.txt et relancez l'application.")
    st.stop()

st.title("Extraction de PDF vers texte")

# Bandeau sous le titre : ligne + site + LinkedIn (logo + lien)
linkedin_svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 34 34" width="14" height="14" aria-hidden="true">
  <g>
    <path fill="#0A66C2" d="M34,17c0,9.4-7.6,17-17,17S0,26.4,0,17S7.6,0,17,0S34,7.6,34,17z"/>
    <path fill="#FFFFFF" d="M10,13h4v11h-4V13z M12,10c-1.3,0-2.3-1-2.3-2.2C9.7,6.5,10.7,5.5,12,5.5S14.3,6.5,14.3,7.8
      C14.3,9,13.3,10,12,10z M18,13h3.8v1.5h0.1c0.5-0.9,1.8-1.9,3.7-1.9c4,0,4.7,2.6,4.7,5.9V24h-4v-4.9c0-1.2,0-2.7-1.6-2.7
      c-1.6,0-1.8,1.3-1.8,2.6V24h-4V13z"/>
  </g>
</svg>
"""
st.markdown(
    """
<hr style="margin-top:0.5rem;margin-bottom:0.75rem;">
<div style="display:flex;flex-direction:column;gap:0.25rem;font-size:0.95rem;">
  <div><a href="https://www.codeandcortex.fr" target="_blank">www.codeandcortex.fr</a></div>
  <div style="display:flex;align-items:center;gap:0.4rem;">
    <span>""" + linkedin_svg + """</span>
    <a href="https://www.linkedin.com/in/st%C3%A9phane-meurisse-27339055/" target="_blank" rel="noopener">
      Profil LinkedIn de Stéphane Meurisse
    </a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write(
    "Cette application utilise uniquement PyMuPDF (fitz). Elle gère les métadonnées et propose une heuristique pour la lecture en 1 ou 2 colonnes."
)

with st.sidebar:
    st.header("Paramètres d'extraction")

    mode_colonnes = st.selectbox(
        "Gestion des colonnes",
        options=["auto", "1", "2"],
        index=0,
        help="Heuristique colonnes basée sur les blocs PyMuPDF."
    )

    with st.expander("Métadonnées à inclure dans le .txt"):
        inclure_meta = st.checkbox("Inclure les métadonnées en tête du .txt", value=True)
        champs_possibles = ["Titre", "Auteur", "Sujet", "Mots-clés", "Créateur", "Producteur", "Créé le", "Modifié le"]
        champs_meta_selectionnes = st.multiselect(
            "Champs à inclure",
            options=champs_possibles,
            default=["Titre", "Auteur", "Sujet", "Mots-clés"]
        )

    with st.expander("Nettoyage minimal du texte"):
        enlever_doubles_espaces = st.checkbox("Réduire les espacements multiples", value=True)
        enlever_lignes_vides_multiples = st.checkbox("Compacter les lignes vides successives", value=True)

fichiers = st.file_uploader("Déposez un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True)

resultats = []

if fichiers:
    col_g, col_d = st.columns([1, 1])
    with col_g:
        st.subheader("Aperçu et métadonnées")
    with col_d:
        st.subheader("Téléchargements")

    buffer_zip = io.BytesIO()
    zf = zipfile.ZipFile(buffer_zip, mode="w", compression=zipfile.ZIP_DEFLATED)

    for fichier in fichiers:
        nom = fichier.name
        data = fichier.read()

        # Avertissement si le PDF semble être un scan sans texte
        try:
            if detecter_si_pdf_scanné(data):
                st.warning(f"{nom} semble contenir très peu de texte extractible. "
                           f"S'il s'agit d'un scan, l'extraction peut être incomplète (pas d'OCR dans cette application).")
        except Exception:
            pass

        # Métadonnées
        try:
            meta = lire_metadonnees_pdf(data)
        except Exception as e:
            meta = {
                "Titre": "", "Auteur": "", "Sujet": "", "Mots-clés": "",
                "Créateur": "", "Producteur": "", "Créé le": "", "Modifié le": ""
            }
            st.info(f"Métadonnées non lues pour {nom} : {e}")

        # Extraction PyMuPDF
        erreur = None
        texte = ""
        try:
            texte = extraire_texte_pymupdf(data, mode_colonnes=mode_colonnes)
        except Exception as e:
            erreur = str(e)

        # Nettoyage minimal
        if not erreur and texte:
            if enlever_doubles_espaces:
                import re
                texte = re.sub(r"[ \t]{2,}", " ", texte)
            if enlever_lignes_vides_multiples:
                import re
                texte = re.sub(r"\n{3,}", "\n\n", texte)

        # Assemblage final
        if not erreur:
            txt_final = formater_sortie_texte(nom, texte, inclure_meta, meta, champs_meta_selectionnes)
            contenu_bytes = txt_final.encode("utf-8", errors="ignore")
        else:
            txt_final = f"Erreur d'extraction pour {nom} :\n{erreur}"
            contenu_bytes = txt_final.encode("utf-8", errors="ignore")

        # UI par fichier
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.expander(f"Métadonnées : {nom}", expanded=False):
                for k, v in meta.items():
                    st.write(f"**{k}** : {v if v else '—'}")
            with st.expander(f"Aperçu du texte : {nom}", expanded=False):
                ap = txt_final[:4000]
                if len(txt_final) > 4000:
                    ap += "\n[...] (aperçu tronqué)"
                st.code(ap, language="text")

        with col2:
            st.download_button(
                label=f"Télécharger .txt pour {nom}",
                data=contenu_bytes,
                file_name=Path(nom).with_suffix(".txt").name,
                mime="text/plain"
            )

        # Ajout au ZIP
        zf.writestr(Path(nom).with_suffix(".txt").name, contenu_bytes)

        # Récapitulatif
        resultats.append((nom, "PyMuPDF (blocs/colonnes)", erreur is None, len(contenu_bytes)))

    zf.close()
    st.download_button(
        label="Télécharger tous les .txt en ZIP",
        data=buffer_zip.getvalue(),
        file_name=f"extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )

    st.subheader("Récapitulatif des extractions")
    for nom, moteur_eff, ok, taille in resultats:
        st.write(f"{'OK' if ok else 'Échec'} • {nom} • Moteur : {moteur_eff} • Taille sortie : {taille} octets")


# ------------------------------------------------------------
# Déploiement Streamlit Cloud :
# requirements.txt minimal :
#   streamlit
#   PyMuPDF
#
# Astuce : si l'installation échoue, vérifiez la version de PyMuPDF et la compatibilité de l'image Python.
# ------------------------------------------------------------
