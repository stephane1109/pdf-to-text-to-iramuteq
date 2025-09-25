# app.py
# ------------------------------------------------------------
# Application Streamlit pour extraire le texte de PDF et l'exporter en .txt
# PyMuPDF (fitz) uniquement
# - Métadonnées (affichage, sélection et injection dans la sortie .txt)
# - Gestion des colonnes (auto / 1 / 2) avec explication intégrée
# - Variables étoilées : insertion en tête du .txt d'une ligne "****" puis de paires définies par l'utilisateur
# - Nettoyage : espaces, lignes vides, césures, numéros de page, en-têtes/pieds répétés
# - Export individuel et en archive ZIP
# - Bandeau sous le titre : ligne + www.codeandcortex.fr
# - Mise en page large (wide) sur toute l’application
# ------------------------------------------------------------

import io
from pathlib import Path
from datetime import datetime
import zipfile
from collections import Counter

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


def extraire_pages_pymupdf(pdf_bytes: bytes, mode_colonnes: str = "auto") -> list:
    """Extraire le texte page par page via PyMuPDF avec heuristique de colonnes ('auto' | '1' | '2').
    Retourne une liste contenant le texte de chaque page.
    """
    textes_par_page = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            try:
                blocs_raw = page.get_text("blocks")
            except Exception:
                textes_par_page.append(page.get_text("text") or "")
                continue

            blocs = []
            for b in blocs_raw:
                if len(b) >= 5 and isinstance(b[4], str):
                    blocs.append(
                        {"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "text": (b[4] or "").strip()}
                    )

            blocs = [b for b in blocs if b["text"]]
            if not blocs:
                textes_par_page.append(page.get_text("text") or "")
                continue

            largeur = page.rect.width
            milieu = largeur / 2.0

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

            textes_par_page.append(texte_page)
    return textes_par_page


def nettoyer_pieds_entetes_repetes(textes_par_page: list, seuil_ratio: float = 0.6) -> list:
    """Supprimer les en-têtes et pieds de page répétés.
    Heuristique : on regarde la 1ère ligne non vide et la dernière ligne non vide de chaque page,
    et on retire celles qui apparaissent sur au moins 'seuil_ratio' des pages.
    """
    def premiere_ligne_non_vide(s: str) -> str:
        for l in s.splitlines():
            if l.strip():
                return l.strip()
        return ""

    def derniere_ligne_non_vide(s: str) -> str:
        for l in reversed(s.splitlines()):
            if l.strip():
                return l.strip()
        return ""

    premieres = [premiere_ligne_non_vide(p) for p in textes_par_page]
    dernieres = [derniere_ligne_non_vide(p) for p in textes_par_page]

    c_top = Counter([l for l in premieres if l])
    c_bot = Counter([l for l in dernieres if l])

    n = len(textes_par_page)
    top_a_retirer = {l for l, c in c_top.items() if c >= max(2, int(seuil_ratio * n))}
    bot_a_retirer = {l for l, c in c_bot.items() if c >= max(2, int(seuil_ratio * n))}

    nettoyees = []
    for p in textes_par_page:
        lignes = p.splitlines()
        # Retrait haut
        while lignes and lignes[0].strip() in top_a_retirer:
            lignes.pop(0)
        # Retrait bas
        while lignes and lignes[-1].strip() in bot_a_retirer:
            lignes.pop()
        nettoyees.append("\n".join(lignes))
    return nettoyees


def supprimer_numeros_de_page_isoles(textes_par_page: list) -> list:
    """Supprimer les lignes qui ne contiennent qu'un numéro (ex: '12') au début/fin de page."""
    import re
    motif = re.compile(r"^\s*\d+\s*$")
    nettoyees = []
    for p in textes_par_page:
        lignes = p.splitlines()
        # Début
        while lignes and motif.match(lignes[0]):
            lignes.pop(0)
        # Fin
        while lignes and motif.match(lignes[-1]):
            lignes.pop()
        nettoyees.append("\n".join(lignes))
    return nettoyees


def supprimer_cesures_en_fin_de_ligne(texte: str) -> str:
    """Supprimer les césures en fin de ligne (mot coupé par un tiret suivi d'un retour)."""
    import re
    return re.sub(r"-\n(\S)", r"\1", texte)


def appliquer_nettoyages(textes_par_page: list,
                         enlever_doubles_espaces: bool,
                         compacter_lignes_vides: bool,
                         enlever_cesures: bool,
                         enlever_num_pages: bool,
                         enlever_entetes_pieds: bool) -> str:
    """Appliquer les nettoyages demandés et retourner un texte global."""
    pages = textes_par_page[:]

    if enlever_entetes_pieds:
        pages = nettoyer_pieds_entetes_repetes(pages, seuil_ratio=0.6)
    if enlever_num_pages:
        pages = supprimer_numeros_de_page_isoles(pages)

    texte = "\n\n".join(pages)

    if enlever_cesures:
        texte = supprimer_cesures_en_fin_de_ligne(texte)

    if enlever_doubles_espaces:
        import re
        texte = re.sub(r"[ \t]{2,}", " ", texte)

    if compacter_lignes_vides:
        import re
        texte = re.sub(r"\n{3,}", "\n\n", texte)

    return texte


def construire_entete_variables_etoilees(activer: bool, saisie: str) -> str:
    """Construire la section en tête avec '****' suivi des variables étoilées utilisateur.
    Format de saisie attendu (une variable par ligne) : nom=valeur
    Rend :
    ****
    *nom* : valeur
    *nom2* : valeur2
    """
    if not activer:
        return ""
    lignes = []
    lignes.append("****")
    for brut in saisie.splitlines():
        if "=" in brut:
            nom, val = brut.split("=", 1)
            nom = nom.strip()
            val = val.strip()
            if nom:
                lignes.append(f"*{nom}* : {val}")
        elif brut.strip():
            # Si l'utilisateur donne juste un mot, on l'insère avec valeur vide
            lignes.append(f"*{brut.strip()}* :")
    lignes.append("-" * 60)
    return "\n".join(lignes) + "\n"


def formater_sortie_texte(nom_fichier: str,
                          entete_vars: str,
                          texte: str,
                          inclure_meta: bool,
                          meta: dict,
                          champs_meta_selectionnes: list) -> str:
    """Assembler le texte final : variables étoilées, métadonnées (optionnelles), puis corps."""
    parties = []

    if entete_vars.strip():
        parties.append(entete_vars.rstrip("\n"))

    if inclure_meta:
        header = []
        header.append(f"Fichier source : {nom_fichier}")
        for champ in champs_meta_selectionnes:
            if champ in meta and str(meta[champ]).strip():
                header.append(f"{champ} : {meta[champ]}")
        header.append(f"Date d'extraction : {datetime.now().isoformat(timespec='seconds')}")
        header.append("-" * 60)
        parties.append("\n".join(header))

    parties.append(texte or "")
    return "\n\n".join([p for p in parties if p is not None])


# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------

st.set_page_config(page_title="Extraction PDF → Texte", page_icon="📄", layout="wide")

if fitz is None:
    st.error("PyMuPDF (fitz) est requis mais indisponible. Ajoutez 'PyMuPDF' dans requirements.txt et relancez l'application.")
    st.stop()

st.title("Extraction de PDF vers texte")

# Bandeau sous le titre : ligne + site (sans LinkedIn)
st.markdown(
    """
<hr style="margin-top:0.5rem;margin-bottom:0.75rem;">
<div style="display:flex;flex-direction:column;gap:0.25rem;font-size:0.95rem;">
  <div><a href="https://www.codeandcortex.fr" target="_blank">www.codeandcortex.fr</a></div>
</div>
""",
    unsafe_allow_html=True,
)

st.write(
    "Cette application utilise uniquement PyMuPDF (fitz). Elle gère les métadonnées, propose une lecture en 1 ou 2 colonnes, "
    "et permet d'insérer des variables étoilées en tête du texte, ainsi que plusieurs options de nettoyage."
)

# Barre latérale : tous les réglages
with st.sidebar:
    st.header("Paramètres d'extraction")

    mode_colonnes = st.selectbox(
        "Gestion des colonnes",
        options=["auto", "1", "2"],
        index=0,
        help="Sélection du mode de lecture des colonnes."
    )
    st.caption("Explication : 'auto' choisit automatiquement. '1' = lecture de gauche à droite de haut en bas. '2' = deux colonnes : colonne gauche puis colonne droite.")

    with st.expander("Variables étoilées (en tête du .txt)"):
        activer_vars = st.checkbox("Activer l'insertion de variables étoilées", value=False)
        saisie_vars = st.text_area(
            "Saisissez une variable par ligne au format nom=valeur",
            value="",
            height=120,
            help="Exemple :\nsource=JO officiel\nprojet=loi IA"
        )

    with st.expander("Métadonnées à inclure dans le .txt"):
        inclure_meta = st.checkbox("Inclure les métadonnées en tête du .txt", value=True)
        champs_possibles = ["Titre", "Auteur", "Sujet", "Mots-clés", "Créateur", "Producteur", "Créé le", "Modifié le"]
        champs_meta_selectionnes = st.multiselect(
            "Champs à inclure",
            options=champs_possibles,
            default=["Titre", "Auteur", "Sujet", "Mots-clés"]
        )

    with st.expander("Nettoyage du texte"):
        enlever_doubles_espaces = st.checkbox("Réduire les espacements multiples", value=True)
        compacter_lignes_vides = st.checkbox("Compacter les lignes vides successives", value=True)
        enlever_cesures = st.checkbox("Supprimer les césures en fin de ligne", value=True)
        enlever_num_pages = st.checkbox("Supprimer les numéros de page isolés", value=True)
        enlever_entetes_pieds = st.checkbox("Supprimer en-têtes et pieds répétés", value=True)

# Zone centrale, en large : upload multi-fichiers et résultats
fichiers = st.file_uploader("Déposez un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True)

resultats = []

if fichiers:
    st.subheader("Traitement et téléchargements")

    buffer_zip = io.BytesIO()
    zf = zipfile.ZipFile(buffer_zip, mode="w", compression=zipfile.ZIP_DEFLATED)

    for fichier in fichiers:
        nom = fichier.name
        data = fichier.read()

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

        # Extraction PyMuPDF page à page
        erreur = None
        try:
            pages = extraire_pages_pymupdf(data, mode_colonnes=mode_colonnes)
        except Exception as e:
            pages = []
            erreur = str(e)

        # Nettoyages
        texte_global = ""
        if not erreur:
            texte_global = appliquer_nettoyages(
                textes_par_page=pages,
                enlever_doubles_espaces=enlever_doubles_espaces,
                compacter_lignes_vides=compacter_lignes_vides,
                enlever_cesures=enlever_cesures,
                enlever_num_pages=enlever_num_pages,
                enlever_entetes_pieds=enlever_entetes_pieds
            )

        # Variables étoilées en tête
        entete_vars = construire_entete_variables_etoilees(activer_vars, saisie_vars)

        # Assemblage final
        if not erreur:
            txt_final = formater_sortie_texte(
                nom_fichier=nom,
                entete_vars=entete_vars,
                texte=texte_global,
                inclure_meta=inclure_meta,
                meta=meta,
                champs_meta_selectionnes=champs_meta_selectionnes
            )
            contenu_bytes = txt_final.encode("utf-8", errors="ignore")
        else:
            txt_final = f"Erreur d'extraction pour {nom} :\n{erreur}"
            contenu_bytes = txt_final.encode("utf-8", errors="ignore")

        # Aperçu en plein large
        with st.expander(f"Aperçu et métadonnées : {nom}", expanded=False):
            with st.container():
                st.markdown("**Métadonnées détectées**")
                for k, v in meta.items():
                    st.write(f"**{k}** : {v if v else '—'}")
                st.markdown("---")
                st.markdown("**Aperçu du .txt**")
                ap = txt_final[:10000]
                if len(txt_final) > 10000:
                    ap += "\n[...] (aperçu tronqué)"
                st.code(ap, language="text")

        # Téléchargement individuel
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
# Remarques :
# - L'application est en 'wide' via st.set_page_config(layout='wide') et l'interface centrale n'utilise pas de colonnes étroites.
# - L'option colonnes : 'auto' décide tout seul ; '1' = lecture séquentielle des blocs ; '2' = deux colonnes (gauche puis droite).
# ------------------------------------------------------------
