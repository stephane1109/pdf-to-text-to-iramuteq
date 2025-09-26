# app.py
# ------------------------------------------------------------
# Extraction PDF -> .txt (PyMuPDF uniquement)
# Largeur 'wide', bandeau sous le titre (site), multi-fichiers
# Variables étoilées sur UNE LIGNE : "**** *var1 *var2 ..." (espaces -> "-_")
# Métadonnées (affichage + injection optionnelle)
# Gestion des colonnes : auto / 1 / 2
# Portée d'extraction colonnes : totale / gauche uniquement / droite uniquement
# Intervalle de pages (ex : 3 à 98)
# Réparation des espaces intra-mot / ligatures (reconstruction par mots, seuil réglable)
# Nettoyages : espaces multiples, lignes vides, césures (incl. soft hyphen), numéros de page,
#              en-têtes/pieds répétés (détection multi-lignes) + motifs regex personnalisés robustes aux espaces/accents
# Option : passer tout le texte en minuscules
# Export individuel et ZIP
# ------------------------------------------------------------

import io
from pathlib import Path
from datetime import datetime
import zipfile
from collections import Counter, defaultdict
import unicodedata
import re
import streamlit as st

# Dépendance obligatoire
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# ------------------------------------------------------------
# Utilitaires : métadonnées, heuristiques scan
# ------------------------------------------------------------

def lire_metadonnees_pdf(pdf_bytes: bytes) -> dict:
    """Lire les métadonnées du PDF avec PyMuPDF et retourner un dict plat."""
    meta = {
        "Titre": "", "Auteur": "", "Sujet": "", "Mots-clés": "",
        "Créateur": "", "Producteur": "", "Créé le": "", "Modifié le": "",
    }
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        md = doc.metadata or {}
        correspondance = {
            "title": "Titre", "author": "Auteur", "subject": "Sujet",
            "keywords": "Mots-clés", "creator": "Créateur", "producer": "Producteur",
            "creationDate": "Créé le", "modDate": "Modifié le",
        }
        for k_src, k_dst in correspondance.items():
            if k_src in md and md[k_src]:
                meta[k_dst] = str(md[k_src])
    return meta


def detecter_si_pdf_scanné(pdf_bytes: bytes, pages_test: int = 3, seuil_caractères: int = 40) -> bool:
    """Heuristique simple : très peu de texte extractible => probablement un scan."""
    total = 0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        nb = min(len(doc), pages_test)
        for i in range(nb):
            t = doc[i].get_text("text") or ""
            total += len(t.strip())
    return total < seuil_caractères


# ------------------------------------------------------------
# Reconstruction par mots (corrige 'infl uence')
# ------------------------------------------------------------

def _reconstruire_lignes_par_mots(words, seuil_jointure_pts: float = 1.0) -> str:
    """Reconstruire le texte d'une page à partir de 'words' en recollant les segments trop proches."""
    lignes = defaultdict(list)
    for (x0, y0, x1, y1, w, bno, lno, wno) in words:
        lignes[(bno, lno)].append((x0, x1, w))

    lignes_ordonnees = []
    for key in sorted(lignes.keys()):
        mots = sorted(lignes[key], key=lambda t: t[0])
        if not mots:
            continue
        morceaux = [mots[0][2]]
        prev_x1 = mots[0][1]
        for x0, x1, w in mots[1:]:
            gap = x0 - prev_x1
            if gap < seuil_jointure_pts:
                morceaux[-1] = morceaux[-1] + w
            else:
                morceaux.append(w)
            prev_x1 = x1
        lignes_ordonnees.append(" ".join(morceaux))

    return "\n".join(lignes_ordonnees)


# ------------------------------------------------------------
# Gestion colonnes et portée
# ------------------------------------------------------------

def _filtrer_par_portee_words(words, milieu: float, portee_colonnes: str):
    if portee_colonnes == "totale":
        return words
    if portee_colonnes == "gauche":
        return [w for w in words if w[2] <= milieu]  # x1 <= milieu
    if portee_colonnes == "droite":
        return [w for w in words if w[0] >= milieu]  # x0 >= milieu
    return words


def _filtrer_par_portee_blocs(blocs, milieu: float, portee_colonnes: str):
    if portee_colonnes == "totale":
        return blocs
    if portee_colonnes == "gauche":
        return [b for b in blocs if b["x1"] <= milieu]
    if portee_colonnes == "droite":
        return [b for b in blocs if b["x0"] >= milieu]
    return blocs


def extraire_pages_pymupdf(pdf_bytes: bytes,
                           mode_colonnes: str,
                           portee_colonnes: str,
                           reparer_ligatures: bool,
                           seuil_jointure_pts: float,
                           page_min_1based: int = 1,
                           page_max_1based: int = 10**9) -> list:
    """Extraction page par page avec options colonnes, portée et reconstruction par mots."""
    textes_par_page = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        n = len(doc)
        pmin = max(1, page_min_1based)
        pmax = min(n, page_max_1based)
        if pmin > pmax:
            return []

        for i in range(pmin - 1, pmax):
            page = doc[i]
            largeur = page.rect.width
            milieu = largeur / 2.0

            if reparer_ligatures:
                words = page.get_text("words") or []
                words = _filtrer_par_portee_words(words, milieu, portee_colonnes)
                texte_page = _reconstruire_lignes_par_mots(words, seuil_jointure_pts=seuil_jointure_pts)
                textes_par_page.append(texte_page)
                continue

            try:
                blocs_raw = page.get_text("blocks")
            except Exception:
                textes_par_page.append(page.get_text("text") or "")
                continue

            blocs = []
            for b in blocs_raw:
                if len(b) >= 5 and isinstance(b[4], str):
                    blocs.append({"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "text": (b[4] or "").strip()})
            blocs = [b for b in blocs if b["text"]]
            if not blocs:
                textes_par_page.append(page.get_text("text") or "")
                continue

            blocs = _filtrer_par_portee_blocs(blocs, milieu, portee_colonnes)
            if not blocs:
                textes_par_page.append("")
                continue

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


# ------------------------------------------------------------
# Nettoyages dont en-têtes/pieds multi-lignes + regex perso robustes
# ------------------------------------------------------------

def _normaliser_ligne(s: str) -> str:
    """Normaliser pour comparaison : strip, compacter espaces."""
    s = s.replace("\u00AD", "")
    s = " ".join(s.strip().split())
    return s


def _strip_accents(s: str) -> str:
    """Supprimer les accents par normalisation unicode."""
    s_norm = unicodedata.normalize('NFD', s)
    return "".join(ch for ch in s_norm if unicodedata.category(ch) != 'Mn')


def _premieres_k_lignes_non_vides(s: str, k: int) -> list:
    out = []
    for l in s.splitlines():
        if l.strip():
            out.append(l)
            if len(out) >= k:
                break
    return out


def _dernieres_k_lignes_non_vides(s: str, k: int) -> list:
    out = []
    for l in reversed(s.splitlines()):
        if l.strip():
            out.append(l)
            if len(out) >= k:
                break
    out.reverse()
    return out


def nettoyer_entetes_pieds_multilignes(textes_par_page: list,
                                       k_lignes: int = 3,
                                       seuil_ratio: float = 0.6) -> list:
    """Retirer des bandeaux répétés en haut et en bas."""
    n = len(textes_par_page)
    if n == 0:
        return textes_par_page

    top_seqs = Counter()
    bot_seqs = Counter()

    for p in textes_par_page:
        top = _premieres_k_lignes_non_vides(p, k_lignes)
        bot = _dernieres_k_lignes_non_vides(p, k_lignes)
        for L in range(1, k_lignes + 1):
            if len(top) >= L:
                norm = tuple(_normaliser_ligne(x) for x in top[:L])
                if all(norm):
                    top_seqs[norm] += 1
            if len(bot) >= L:
                norm = tuple(_normaliser_ligne(x) for x in bot[-L:])
                if all(norm):
                    bot_seqs[norm] += 1

    seuil_pages = max(2, int(seuil_ratio * n))
    tops_a_retirer = {seq for seq, c in top_seqs.items() if c >= seuil_pages}
    bots_a_retirer = {seq for seq, c in bot_seqs.items() if c >= seuil_pages}

    nettoyees = []
    for p in textes_par_page:
        lignes = p.splitlines()
        # Retrait haut
        for L in range(k_lignes, 0, -1):
            cand = _premieres_k_lignes_non_vides("\n".join(lignes), L)
            norm = tuple(_normaliser_ligne(x) for x in cand[:L])
            if norm and norm in tops_a_retirer:
                to_remove = L
                new_lines = []
                skipped = 0
                for l in lignes:
                    if l.strip() and skipped < to_remove:
                        skipped += 1
                        continue
                    new_lines.append(l)
                lignes = new_lines
                break
        # Retrait bas
        for L in range(k_lignes, 0, -1):
            cand = _dernieres_k_lignes_non_vides("\n".join(lignes), L)
            norm = tuple(_normaliser_ligne(x) for x in cand[-L:])
            if norm and norm in bots_a_retirer:
                to_remove = L
                new_lines = []
                skipped = 0
                for l in reversed(lignes):
                    if l.strip() and skipped < to_remove:
                        skipped += 1
                        continue
                    new_lines.append(l)
                lignes = list(reversed(new_lines))
                break
        nettoyees.append("\n".join(lignes))
    return nettoyees


def supprimer_numeros_de_page_isoles(textes_par_page: list) -> list:
    """Supprimer les lignes ne contenant qu'un nombre en début/fin de page."""
    motif = re.compile(r"^\s*\d+\s*$")
    nettoyees = []
    for p in textes_par_page:
        lignes = p.splitlines()
        while lignes and motif.match(lignes[0]):
            lignes.pop(0)
        while lignes and motif.match(lignes[-1]):
            lignes.pop()
        nettoyees.append("\n".join(lignes))
    return nettoyees


def supprimer_cesures_en_fin_de_ligne(texte: str) -> str:
    """Supprimer les césures et le 'soft hyphen' U+00AD."""
    texte = texte.replace("\u00AD", "")
    return re.sub(r"-\n(\S)", r"\1", texte)


def _ligne_correspond_a_un_motif(ligne: str, compiled_patterns) -> bool:
    """Tester la ligne contre plusieurs variantes normalisées et les regex fournies."""
    s = ligne.strip()
    if not s:
        return False
    variants = []

    # Variantes
    s1 = re.sub(r"\s+", " ", s)               # espaces compactés
    s2 = re.sub(r"\s+", "", s)                # sans espaces
    s3 = _strip_accents(s1)                   # sans accents (espaces compactés)
    s4 = _strip_accents(s2)                   # sans accents et sans espaces

    variants.extend([s, s1, s2, s3, s4])

    # Comparaison en ignorant la casse ; fullmatch sur les variantes
    for cp in compiled_patterns:
        for v in variants:
            if cp.fullmatch(v, pos=0, endpos=len(v)):
                return True
    return False


def appliquer_regex_personnalisees(texte: str, motifs_raw: str) -> str:
    """Appliquer des regex robustes : suppression de lignes qui correspondent aux motifs
    après normalisations (espaces/accents). Un motif par ligne.
    """
    motifs = [m.strip() for m in motifs_raw.splitlines() if m.strip()]
    if not motifs:
        return texte

    compiled = []
    for m in motifs:
        try:
            compiled.append(re.compile(m, flags=re.IGNORECASE))
        except re.error:
            # motif invalide : on l'escapera automatiquement (sécurité)
            compiled.append(re.compile(re.escape(m), flags=re.IGNORECASE))

    res = []
    for l in texte.splitlines():
        if _ligne_correspond_a_un_motif(l, compiled):
            continue
        res.append(l)
    return "\n".join(res)


def appliquer_nettoyages(textes_par_page: list,
                         enlever_doubles_espaces: bool,
                         compacter_lignes_vides: bool,
                         enlever_cesures: bool,
                         enlever_num_pages: bool,
                         enlever_entetes_pieds: bool,
                         k_lignes_ep: int,
                         seuil_ratio_ep: float,
                         motifs_regex_ep: str,
                         to_lower: bool) -> str:
    """Appliquer les nettoyages demandés et retourner un texte global."""
    pages = textes_par_page[:]

    if enlever_entetes_pieds:
        pages = nettoyer_entetes_pieds_multilignes(pages, k_lignes=k_lignes_ep, seuil_ratio=seuil_ratio_ep)
    if enlever_num_pages:
        pages = supprimer_numeros_de_page_isoles(pages)

    texte = "\n\n".join(pages)

    if motifs_regex_ep.strip():
        texte = appliquer_regex_personnalisees(texte, motifs_regex_ep)

    if enlever_cesures:
        texte = supprimer_cesures_en_fin_de_ligne(texte)

    if enlever_doubles_espaces:
        texte = re.sub(r"[ \t]{2,}", " ", texte)

    if compacter_lignes_vides:
        texte = re.sub(r"\n{3,}", "\n\n", texte)

    if to_lower:
        texte = texte.lower()

    return texte


# ------------------------------------------------------------
# Variables étoilées
# ------------------------------------------------------------

def encoder_nom_variable(var: str) -> str:
    """Encoder un nom de variable : trim, supprimer astérisques initiaux, couper à '=', espaces -> '-_'."""
    var = var.strip()
    while var.startswith("*"):
        var = var[1:].lstrip()
    if var and var[-1] in {",", ";", ":"}:
        var = var[:-1].rstrip()
    var = var.split("=", 1)[0].strip()
    var = " ".join(var.split())
    var = var.replace(" ", "-_")
    return var


def construire_entete_variables_etoilees(saisie: str) -> str:
    """Une seule ligne : '**** *var1 *var2 ...' ; vide si aucune variable valide."""
    tokens = []
    for brut in saisie.splitlines():
        brut = brut.strip()
        if not brut:
            continue
        nom = encoder_nom_variable(brut)
        if nom:
            tokens.append(f"*{nom}")
    if not tokens:
        return ""
    return "**** " + " ".join(tokens) + "\n"


def formater_sortie_texte(nom_fichier: str,
                          entete_vars: str,
                          texte: str,
                          inclure_meta: bool,
                          meta: dict,
                          champs_meta_selectionnes: list) -> str:
    """Assembler : variables étoilées (si non vide), métadonnées (optionnelles), corps."""
    parties = []
    if entete_vars:
        parties.append(entete_vars.rstrip("\n"))
    if inclure_meta:
        header = [f"Fichier source : {nom_fichier}"]
        for champ in champs_meta_selectionnes:
            if champ in meta and str(meta[champ]).strip():
                header.append(f"{champ} : {meta[champ]}")
        header.append(f"Date d'extraction : {datetime.now().isoformat(timespec='seconds')}")
        header.append("-" * 60)
        parties.append("\n".join(header))
    parties.append(texte or "")
    return "\n\n".join(parties)


# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------

st.set_page_config(page_title="Extraction PDF → Texte", page_icon="📄", layout="wide")

if fitz is None:
    st.error("PyMuPDF est requis mais indisponible. Ajoutez 'PyMuPDF' dans requirements.txt et relancez l'application.")
    st.stop()

st.title("Extraction PDF to TEXT to IRAMUTEQ")

# Bandeau sous le titre (site)
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
    "Application PyMuPDF. Métadonnées, lecture 1/2 colonnes, sélection de colonnes, "
    "intervalle de pages, variables étoilées en tête, nettoyages, regex..."
)

with st.sidebar:
    st.header("Paramètres d'extraction")

    mode_colonnes = st.selectbox(
        "Gestion des colonnes",
        options=["auto", "1", "2"],
        index=0,
        help="auto choisit automatiquement; '1' = lecture séquentielle; '2' = deux colonnes (gauche puis droite)."
    )
    st.caption("Explication : 'auto' décide tout seul. '1' = haut→bas, gauche→droite. '2' = colonne gauche puis colonne droite.")

    portee_colonnes = st.selectbox(
        "Portée d'extraction (colonnes)",
        options=["Totale", "Colonne gauche uniquement", "Colonne droite uniquement"],
        index=0,
        help="Utile si le PDF à deux colonnes mélange deux langues par page."
    )
    mapping_portee = {"Totale": "totale", "Colonne gauche uniquement": "gauche", "Colonne droite uniquement": "droite"}
    portee_colonnes_key = mapping_portee[portee_colonnes]

    limiter_pages = st.checkbox("Limiter l'extraction à un intervalle de pages", value=False)
    page_debut = st.number_input("Page de début (1-based)", min_value=1, value=1, step=1, disabled=not limiter_pages)
    page_fin = st.number_input("Page de fin (inclusif, 1-based)", min_value=1, value=9999, step=1, disabled=not limiter_pages)

    with st.expander("Réparer espaces intra-mot / ligatures"):
        reparer_ligatures = st.checkbox("Activer la reconstruction par mots (répare 'infl uence')", value=True)
        seuil_jointure_pts = st.number_input(
            "Seuil de jointure (points PDF)",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="Si la distance entre deux 'mots' est < seuil, ils sont recollés sans espace."
        )

    with st.expander("Variables étoilées (en tête du .txt)"):
        saisie_vars = st.text_area(
            "Une variable * par ligne. Espaces entre les mots, le script ajoute -> '-_'.",
            value="",
            height=120,
            help="Exemples :\nprojet loi\nsource=JO officiel\nversion brouillon"
        )
        apercu_entete = construire_entete_variables_etoilees(saisie_vars)
        if apercu_entete:
            st.code(apercu_entete.strip("\n"), language="text")
        else:
            st.caption("Aucun en-tête ne sera ajouté tant qu'aucune variable valide n'est saisie.")

    with st.expander("Métadonnées du pdf à inclure dans le .txt"):
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
        enlever_cesures = st.checkbox("Supprimer les césures", value=True)
        enlever_num_pages = st.checkbox("Supprimer les numéros de page isolés", value=True)

        enlever_entetes_pieds = st.checkbox("Supprimer en-têtes et pieds répétés (ReGex)", value=True)
        k_lignes_ep = st.number_input("Nombre max de lignes à examiner en tête/pied (1-5)", min_value=1, max_value=5, value=3, step=1)
        seuil_ratio_ep = st.slider("Seuil de répétition pour suppression", min_value=0.4, max_value=0.9, value=0.6, step=0.05,
                                   help="Proportion minimale de pages où le motif apparaît pour être supprimé.")

        # Motifs fournis (pré-remplis). Formes compactes, robustes via normalisations (espaces/accents).
        motifs_regex_ep = st.text_area(
            "Motifs regex personnalisés (un par ligne, correspondance sur la ligne entière après normalisations)",
            value=(
                "^CHAMBRE4eSESSIONDELA53eLEGISLATURE$\n"
                "^2012$\n"
                "^DOC532880/0013$"
            ),
            height=120,
            help="Formes compactes pensés pour matcher après normalisations (espaces/accents retirés). "
                 "Ajoutez vos motifs, un par ligne."
        )

        to_lower = st.checkbox("Passer tout le texte en minuscules", value=False)


# Upload multi-fichiers
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
                           f"S'il s'agit d'un scan, l'extraction peut être incomplète (pas d'OCR).")
        except Exception:
            pass

        # Métadonnées
        try:
            meta = lire_metadonnees_pdf(data)
        except Exception as e:
            meta = {k: "" for k in ["Titre", "Auteur", "Sujet", "Mots-clés", "Créateur", "Producteur", "Créé le", "Modifié le"]}
            st.info(f"Métadonnées non lues pour {nom} : {e}")

        # Intervalle de pages
        if limiter_pages:
            with fitz.open(stream=data, filetype="pdf") as doc_tmp:
                n_pages = len(doc_tmp)
            pmin = max(1, int(page_debut))
            pmax = min(n_pages, int(page_fin))
        else:
            with fitz.open(stream=data, filetype="pdf") as doc_tmp:
                pmin, pmax = 1, len(doc_tmp)

        # Extraction
        erreur = None
        try:
            pages = extraire_pages_pymupdf(
                data,
                mode_colonnes=mode_colonnes,
                portee_colonnes=portee_colonnes_key,
                reparer_ligatures=reparer_ligatures,
                seuil_jointure_pts=seuil_jointure_pts,
                page_min_1based=pmin,
                page_max_1based=pmax
            )
        except Exception as e:
            pages = []
            erreur = str(e)

        # Nettoyage global
        texte_global = ""
        if not erreur:
            texte_global = appliquer_nettoyages(
                textes_par_page=pages,
                enlever_doubles_espaces=enlever_doubles_espaces,
                compacter_lignes_vides=compacter_lignes_vides,
                enlever_cesures=enlever_cesures,
                enlever_num_pages=enlever_num_pages,
                enlever_entetes_pieds=enlever_entetes_pieds,
                k_lignes_ep=k_lignes_ep,
                seuil_ratio_ep=seuil_ratio_ep,
                motifs_regex_ep=motifs_regex_ep,
                to_lower=to_lower
            )

        # Variables étoilées
        entete_vars = construire_entete_variables_etoilees(saisie_vars)

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

        # Aperçu
        with st.expander(f"Aperçu et métadonnées : {nom}", expanded=False):
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
        moteur_label = "PyMuPDF (reconstruction par mots)" if reparer_ligatures else f"PyMuPDF (colonnes {mode_colonnes})"
        portee_label = {"totale": "Totale", "gauche": "Colonne gauche", "droite": "Colonne droite"}[portee_colonnes_key]
        intervalle_label = f"pages {pmin}-{pmax}"
        resultats.append((nom, moteur_label, portee_label, intervalle_label, (erreur is None), len(contenu_bytes)))

    zf.close()
    st.download_button(
        label="Télécharger tous les .txt en ZIP",
        data=buffer_zip.getvalue(),
        file_name=f"extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )

    st.subheader("Récapitulatif des extractions")
    for nom, moteur_eff, portee_eff, intervalle_eff, ok, taille in resultats:
        st.write(f"{'OK' if ok else 'Échec'} • {nom} • Moteur : {moteur_eff} • Portée : {portee_eff} • Intervalle : {intervalle_eff} • Taille : {taille} octets")


# ------------------------------------------------------------
# requirements.txt minimal :
#   streamlit
#   PyMuPDF
# ------------------------------------------------------------
