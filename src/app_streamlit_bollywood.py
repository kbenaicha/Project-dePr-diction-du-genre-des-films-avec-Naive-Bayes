from pathlib import Path
import base64 as _b64
import streamlit.components.v1 as _components

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BollywoodAI — Conseil Investissement",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "Bollywood_Movies_data" / "bollywood_clean.csv"
MODELS_DIR = BASE_DIR / "models"

# Features disponibles AVANT tournage (pas de note ni votes)
FEATURES = ["Budget(INR)", "Number of Screens", "Genre_Encoded"]

# ── Palette Bollywood ──────────────────────────────────────
SAFFRON  = "#FF6B00"
CRIMSON  = "#C0151E"
GOLD     = "#FFB800"
MARIGOLD = "#FF8C00"
TURMERIC = "#E8500A"
IVORY    = "#FFF8EC"
DEEP     = "#1A0500"
WARM_MID = "#3D1A00"
BORDER_C = "#6B2F00"
MUTED_C  = "#9B7B5A"
SUCCESS_C= "#2ECC71"
WARN_C   = "#F39C12"
DANGER_C = "#E74C3C"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Lato:wght@300;400;700&display=swap');

.stApp {{
    background-color: {DEEP};
    color: {IVORY};
    font-family: 'Lato', sans-serif;
}}
.stApp::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(circle at 15% 25%, rgba(255,107,0,0.06) 0%, transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(192,21,30,0.07) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(255,184,0,0.03) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}}
h1,h2,h3 {{ font-family: 'Cinzel', serif; color: {GOLD}; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: rgba(26,5,0,0.95);
    border-bottom: 1px solid {BORDER_C};
    padding: 0 1.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    color: {MUTED_C};
    font-family: 'Cinzel', serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 1rem 1.75rem;
    border-bottom: 2px solid transparent;
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {GOLD};
    border-bottom: 2px solid {GOLD};
    background: transparent;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 2rem; }}

div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, #220C00 0%, #2E1200 100%);
    border: 1px solid {BORDER_C};
    border-top: 2px solid {SAFFRON};
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}}
div[data-testid="metric-container"] label {{
    color: {MUTED_C} !important;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {GOLD} !important;
    font-size: 1.75rem;
    font-family: 'Cinzel', serif;
}}

.stSlider > div > div > div > div {{ background: {SAFFRON} !important; }}

.stSelectbox [data-baseweb="select"] > div {{
    background: #220C00;
    border: 1px solid {BORDER_C};
    border-radius: 8px;
    color: {IVORY};
}}
.stSelectbox [data-baseweb="select"] > div:hover {{ border-color: {SAFFRON}; }}

.stButton > button {{
    background: linear-gradient(135deg, {SAFFRON} 0%, {CRIMSON} 100%);
    color: {IVORY};
    border: none;
    border-radius: 8px;
    font-family: 'Cinzel', serif;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.1em;
    padding: 0.8rem 2rem;
    width: 100%;
    text-transform: uppercase;
    box-shadow: 0 4px 20px rgba(255,107,0,0.3);
    transition: opacity 0.2s, transform 0.1s;
}}
.stButton > button:hover {{
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(255,107,0,0.45);
}}
.stButton > button:active {{ transform: translateY(0); }}

.streamlit-expanderHeader {{
    background: #220C00 !important;
    border: 1px solid {BORDER_C} !important;
    border-radius: 8px !important;
    color: {IVORY} !important;
}}
hr {{ border-color: {BORDER_C}; margin: 2rem 0; }}

.bw-header {{
    position: relative;
    background: linear-gradient(160deg, #2A0800 0%, #1A0500 50%, #220D00 100%);
    border-bottom: 1px solid {BORDER_C};
    padding: 0;
    margin: -1rem -1rem 0;
    overflow: hidden;
}}
.bw-header-inner {{
    padding: 2.5rem 3rem 2rem;
    position: relative;
    z-index: 1;
}}
.bw-header::after {{
    content: '❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋ ✦ ❋';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    font-size: 0.6rem;
    color: {GOLD};
    opacity: 0.4;
    letter-spacing: 0.4rem;
    text-align: center;
    padding: 3px 0;
    background: linear-gradient(90deg, transparent, rgba(255,184,0,0.08), transparent);
}}
.bw-title {{
    font-family: 'Cinzel', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: {IVORY};
    line-height: 1.1;
    margin: 0;
}}
.bw-title span {{ color: {GOLD}; }}
.bw-subtitle {{
    font-size: 0.82rem;
    color: {MUTED_C};
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    font-weight: 300;
}}
.kpi-strip {{
    display: flex;
    gap: 2.5rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
    border-top: 1px solid rgba(107,47,0,0.5);
    padding-top: 1.25rem;
}}
.kpi-item {{ text-align: left; }}
.kpi-value {{
    font-family: 'Cinzel', serif;
    font-size: 1.5rem;
    color: {GOLD};
    line-height: 1;
}}
.kpi-label {{
    font-size: 0.7rem;
    color: {MUTED_C};
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 4px;
}}
.kpi-divider {{
    width: 1px;
    background: {BORDER_C};
    align-self: stretch;
    opacity: 0.6;
}}

.section-ornament {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}}
.section-ornament::before {{
    content: '';
    flex: 0 0 32px;
    height: 1px;
    background: {BORDER_C};
}}
.section-ornament::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, {BORDER_C}, transparent);
}}
.section-tag {{
    font-family: 'Cinzel', serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {SAFFRON};
    white-space: nowrap;
}}

.result-card {{
    background: linear-gradient(145deg, #2A0A00 0%, #200800 100%);
    border: 1px solid {BORDER_C};
    border-radius: 14px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.result-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {SAFFRON}, {GOLD}, {CRIMSON});
}}
.result-verdict {{
    font-family: 'Cinzel', serif;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0.5rem 0 0.25rem;
}}
.result-sub {{
    font-size: 0.7rem;
    color: {MUTED_C};
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}
.result-conf {{ font-size: 0.82rem; color: {MUTED_C}; margin-top: 0.5rem; }}

.insight-card {{
    background: linear-gradient(135deg, #2A1200 0%, #1E0C00 100%);
    border: 1px solid {BORDER_C};
    border-left: 4px solid {GOLD};
    border-radius: 0 12px 12px 0;
    padding: 1.1rem 1.4rem;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #D4B896;
    margin: 0.75rem 0;
}}
.insight-title {{
    font-family: 'Cinzel', serif;
    font-size: 0.9rem;
    color: {GOLD};
    margin-bottom: 0.4rem;
    font-weight: 600;
}}

.context-box {{
    background: #1E0A00;
    border: 1px solid {BORDER_C};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: {MUTED_C};
    line-height: 1.7;
    margin-top: 1rem;
}}
.context-box b {{ color: {GOLD}; }}

.empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 380px;
    text-align: center;
}}
.empty-icon {{ font-size: 3.5rem; opacity: 0.2; margin-bottom: 1rem; }}
.empty-text {{
    font-family: 'Cinzel', serif;
    font-size: 1rem;
    color: #6B4A2A;
    margin-bottom: 0.5rem;
}}
.empty-hint {{
    font-size: 0.82rem;
    color: #4A3020;
    line-height: 1.7;
    max-width: 280px;
}}

.v-hit    {{ color: {SUCCESS_C}; }}
.v-avg    {{ color: {WARN_C}; }}
.v-flop   {{ color: {DANGER_C}; }}
.v-high   {{ color: {SUCCESS_C}; }}
.v-mid    {{ color: {WARN_C}; }}
.v-low    {{ color: {DANGER_C}; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def format_inr(value):
    if pd.isna(value): return "N/A"
    if value >= 1_000_000_000: return f"₹{value/1_000_000_000:.1f} Md"
    if value >= 1_000_000:     return f"₹{value/1_000_000:.0f} M"
    if value >= 1_000:         return f"₹{value/1_000:.0f} K"
    return f"₹{value:.0f}"


def verdict_css(label):
    return {
        "Hit": "v-hit", "Average": "v-avg", "Flop": "v-flop",
        "High ROI": "v-high", "Medium ROI": "v-mid", "Low ROI": "v-low",
    }.get(label, "")


def interpret_combination(success, profit):
    combos = {
        ("Hit",     "High ROI"):   ("🏆 Investissement d'exception",   "Ce profil cumule forte affluence et excellentes marges. Le scénario idéal — à prioriser absolument."),
        ("Hit",     "Medium ROI"): ("🌟 Grand succès, marges correctes","Forte audience assurée. Les charges absorbent une partie du retour. Investissement solide."),
        ("Hit",     "Low ROI"):    ("⚠️ Succès public, retour faible",  "Film populaire mais budget probablement disproportionné. Revoir les coûts de production."),
        ("Average", "High ROI"):   ("💰 Niche rentable",                "Audience modeste mais coûts maîtrisés — excellente rentabilité. Stratégie de niche efficace."),
        ("Average", "Medium ROI"): ("➡️ Profil standard du marché",     "Performance dans la moyenne Bollywood. Risque limité, rendement limité."),
        ("Average", "Low ROI"):    ("🔻 Risque modéré",                 "Audience correcte mais rentabilité préoccupante. Revoir la stratégie de distribution."),
        ("Flop",    "High ROI"):   ("🔄 Décevant mais rentable",        "Faible audience, budget très contenu — film rentable malgré tout. Rare mais possible."),
        ("Flop",    "Medium ROI"): ("🔻 Sous-performance attendue",     "Film peu vu, retour marginal. Des ajustements significatifs sont nécessaires."),
        ("Flop",    "Low ROI"):    ("❌ Investissement déconseillé",     "Ce profil cumule faible audience et mauvaise rentabilité. À éviter en l'état."),
    }
    return combos.get((success, profit), ("—", "Combinaison non référencée."))


def style_axes_bw(ax, fig=None):
    bg = "#1E0800"
    if fig:
        fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", alpha=0.1, linestyle="--", color="#8B4513")
    ax.tick_params(colors=MUTED_C)
    ax.title.set_color(IVORY)
    ax.xaxis.label.set_color(MUTED_C)
    ax.yaxis.label.set_color(MUTED_C)


def build_prediction_frame(budget_inr, screens, genre_enc):
    return pd.DataFrame([{
        "Budget(INR)":       budget_inr,
        "Number of Screens": screens,
        "Genre_Encoded":     genre_enc,
    }])


# ══════════════════════════════════════════════════════════
# DATA & MODELS
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    # La fusion des genres (love_story/rom_com → romance) est faite dans NB1
    df   = pd.read_csv(DATA_PATH)
    data = df[["Movie Name", "Genre", "Budget(INR)", "Number of Screens",
               "Rating(10)", "Votes", "Revenue(INR)", "ROI"]].copy()
    rev_q33 = data["Revenue(INR)"].quantile(0.33)
    rev_q67 = data["Revenue(INR)"].quantile(0.67)
    roi_q33 = data["ROI"].quantile(0.33)
    roi_q67 = data["ROI"].quantile(0.67)
    data["Success_Class"]       = data["Revenue(INR)"].apply(
        lambda v: "Flop" if v < rev_q33 else ("Average" if v < rev_q67 else "Hit"))
    data["Profitability_Class"] = data["ROI"].apply(
        lambda v: "Low ROI" if v < roi_q33 else ("Medium ROI" if v < roi_q67 else "High ROI"))
    return data


def _try_load_pkl_models():
    needed = ["rf_success.pkl", "gnb_success.pkl", "rf_profit.pkl",
              "gnb_profit.pkl", "le_genre.pkl", "features.json"]
    if not all((MODELS_DIR / f).exists() for f in needed):
        return None
    with open(MODELS_DIR / "features.json") as f:
        pkl_features = json.load(f)
    if pkl_features != FEATURES:
        return None
    le    = joblib.load(MODELS_DIR / "le_genre.pkl")
    rf_s  = joblib.load(MODELS_DIR / "rf_success.pkl")
    gnb_s = joblib.load(MODELS_DIR / "gnb_success.pkl")
    rf_p  = joblib.load(MODELS_DIR / "rf_profit.pkl")
    gnb_p = joblib.load(MODELS_DIR / "gnb_profit.pkl")
    return le, {"Random Forest": rf_s, "Naive Bayes": gnb_s}, \
               {"Random Forest": rf_p, "Naive Bayes": gnb_p}


@st.cache_resource
def train_models():
    with st.spinner("✨  Chargement de la base Bollywood…"):
        raw = pd.read_csv(DATA_PATH)
        # La fusion des genres est faite en amont dans le Notebook 1

        pkl = _try_load_pkl_models()
        if pkl:
            le, success_models, profit_models = pkl
            raw["Genre_Encoded"] = le.transform(raw["Genre"])
            source = "pkl"
        else:
            le = LabelEncoder()
            raw["Genre_Encoded"] = le.fit_transform(raw["Genre"])
            source = "train"

        rev_q33 = raw["Revenue(INR)"].quantile(0.33)
        rev_q67 = raw["Revenue(INR)"].quantile(0.67)
        roi_q33 = raw["ROI"].quantile(0.33)
        roi_q67 = raw["ROI"].quantile(0.67)
        raw["Success_Class"]       = raw["Revenue(INR)"].apply(
            lambda v: "Flop" if v < rev_q33 else ("Average" if v < rev_q67 else "Hit"))
        raw["Profitability_Class"] = raw["ROI"].apply(
            lambda v: "Low ROI" if v < roi_q33 else ("Medium ROI" if v < roi_q67 else "High ROI"))

        X         = raw[FEATURES]
        y_success = raw["Success_Class"]
        y_profit  = raw["Profitability_Class"]

        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
            X, y_success, test_size=0.2, random_state=42, stratify=y_success)
        X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
            X, y_profit,  test_size=0.2, random_state=42, stratify=y_profit)

        def make_gnb():
            # var_smoothing élevé pour éviter la dégénérescence 0%/100%
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("m", GaussianNB(var_smoothing=1e-2))])
        def make_rf():
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("m", RandomForestClassifier(n_estimators=300, random_state=42,
                                                          class_weight="balanced"))])

        if source == "train":
            success_models = {"Naive Bayes": make_gnb(), "Random Forest": make_rf()}
            profit_models  = {"Naive Bayes": make_gnb(), "Random Forest": make_rf()}
            for nm in success_models:
                success_models[nm].fit(X_tr_s, y_tr_s)
                profit_models[nm].fit(X_tr_p, y_tr_p)

        metrics = {}
        for nm in success_models:
            yps = success_models[nm].predict(X_te_s)
            ypp = profit_models[nm].predict(X_te_p)
            ps, rs, f1s, _ = precision_recall_fscore_support(y_te_s, yps, average="macro", zero_division=0)
            pp, rp, f1p, _ = precision_recall_fscore_support(y_te_p, ypp, average="macro", zero_division=0)
            metrics[nm] = {
                "success_accuracy":  accuracy_score(y_te_s, yps),
                "success_precision": ps, "success_recall": rs, "success_f1": f1s,
                "profit_accuracy":   accuracy_score(y_te_p, ypp),
                "profit_precision":  pp, "profit_recall":  rp, "profit_f1":  f1p,
                "success_cm":     confusion_matrix(y_te_s, yps, labels=sorted(y_success.unique())),
                "profit_cm":      confusion_matrix(y_te_p, ypp, labels=sorted(y_profit.unique())),
                "success_labels": sorted(y_success.unique()),
                "profit_labels":  sorted(y_profit.unique()),
            }

        fitted = {}
        for nm in success_models:
            if source == "pkl":
                fitted[nm] = {"success": success_models[nm], "profit": profit_models[nm]}
            else:
                ms = make_rf() if nm == "Random Forest" else make_gnb()
                mp = make_rf() if nm == "Random Forest" else make_gnb()
                fitted[nm] = {"success": ms.fit(X, y_success), "profit": mp.fit(X, y_profit)}

    return load_data(), le, metrics, fitted


data, genre_encoder, metrics, fitted_models = train_models()
best_model    = max(metrics, key=lambda n: metrics[n]["success_f1"])
top_genre     = data.groupby("Genre")["Revenue(INR)"].mean().idxmax()
median_budget = data["Budget(INR)"].median()
n_films       = len(data)


# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="bw-header">
  <div class="bw-header-inner">
    <div class="bw-subtitle">✦ Plateforme d'analyse cinématographique ✦</div>
    <p class="bw-title">Bollywood<span>AI</span></p>
    <div class="bw-subtitle" style="margin-top:0.15rem">Conseil aux investisseurs — prédiction de succès &amp; rentabilité</div>
    <div class="kpi-strip">
      <div class="kpi-item">
        <div class="kpi-value">{n_films:,}</div>
        <div class="kpi-label">Films analysés</div>
      </div>
      <div class="kpi-divider"></div>
      <div class="kpi-item">
        <div class="kpi-value">{top_genre.capitalize()}</div>
        <div class="kpi-label">Genre le plus rentable</div>
      </div>
      <div class="kpi-divider"></div>
      <div class="kpi-item">
        <div class="kpi-value">{format_inr(median_budget)}</div>
        <div class="kpi-label">Budget médian</div>
      </div>
      <div class="kpi-divider"></div>
      <div class="kpi-item">
        <div class="kpi-value">{metrics[best_model]['success_f1']:.0%}</div>
        <div class="kpi-label">Meilleur F1-score</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════
tab_predict, tab_models = st.tabs([
    "🎬  Analyse de projet",
    "📊  Performance des modèles",
])


# ══════════════════════════════════════════════════════════
# ONGLET 1 — ANALYSE PRODUCTEUR
# ══════════════════════════════════════════════════════════
with tab_predict:

    st.markdown(f"""
    <div class="section-ornament"><span class="section-tag">Paramètres du projet</span></div>
    <p style="color:{MUTED_C};font-size:0.88rem;margin-bottom:1.75rem">
        Renseignez les caractéristiques connues <em>avant le début de la production</em>.
        Le modèle s'appuie sur {n_films} films Bollywood référencés.
    </p>
    """, unsafe_allow_html=True)

    col_form, col_space, col_result = st.columns([1.05, 0.08, 1])

    with col_form:
        genre_list = sorted(data["Genre"].unique().tolist())
        genre_sel  = st.selectbox(
            "Genre du film", genre_list,
            index=genre_list.index("comedy") if "comedy" in genre_list else 0,
        )

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        budget_m = st.slider(
            "Budget de production",
            min_value=1, max_value=8000, value=200, step=10,
            format="%d M INR",
        )
        st.caption(f"≈ {format_inr(budget_m * 1_000_000)}")

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        screens = st.slider(
            "Nombre d'écrans de diffusion",
            min_value=1, max_value=5000, value=500, step=50,
        )
        st.caption(f"{screens:,} écrans prévus")

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        model_sel = st.selectbox(
            "Algorithme",
            list(fitted_models.keys()),
            help="Random Forest offre généralement la meilleure précision"
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        predict_btn = st.button("✦  Analyser ce projet  ✦", use_container_width=True)

        nb_genre   = len(data[data["Genre"] == genre_sel])
        nb_similar = len(data[
            (data["Budget(INR)"] >= budget_m * 1e6 * 0.7) &
            (data["Budget(INR)"] <= budget_m * 1e6 * 1.3)
        ])
        st.markdown(f"""
        <div class="context-box">
            Genre <b>{genre_sel}</b> : <b>{nb_genre}</b> films référencés<br>
            Budget comparable (±30%) : <b>{nb_similar}</b> films dans la base
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        if predict_btn:
            try:
                genre_enc = int(genre_encoder.transform([genre_sel])[0])
                X_pred    = build_prediction_frame(budget_m * 1_000_000, screens, genre_enc)

                ms = fitted_models[model_sel]["success"]
                mp = fitted_models[model_sel]["profit"]

                pred_s   = ms.predict(X_pred)[0]
                proba_s  = ms.predict_proba(X_pred)[0]
                labels_s = list(ms.classes_)
                pred_p   = mp.predict(X_pred)[0]
                proba_p  = mp.predict_proba(X_pred)[0]
                labels_p = list(mp.classes_)

                conf_s = max(proba_s)
                conf_p = max(proba_p)
                cs = verdict_css(pred_s)
                cp = verdict_css(pred_p)
                verdict_title, verdict_text = interpret_combination(pred_s, pred_p)

                st.markdown(f"""
                <div class="section-ornament"><span class="section-tag">Résultat de l'analyse</span></div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-bottom:1rem">
                  <div class="result-card">
                    <div class="result-sub">Succès public</div>
                    <div class="result-verdict {cs}">{pred_s}</div>
                    <div class="result-conf">Confiance {conf_s:.0%}</div>
                  </div>
                  <div class="result-card">
                    <div class="result-sub">Rentabilité ROI</div>
                    <div class="result-verdict {cp}">{pred_p}</div>
                    <div class="result-conf">Confiance {conf_p:.0%}</div>
                  </div>
                </div>
                <div class="insight-card">
                  <div class="insight-title">{verdict_title}</div>
                  {verdict_text}
                </div>
                """, unsafe_allow_html=True)

                # Probabilités
                fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
                color_map = {
                    "Flop": DANGER_C, "Average": WARN_C, "Hit": SUCCESS_C,
                    "Low ROI": DANGER_C, "Medium ROI": WARN_C, "High ROI": SUCCESS_C,
                }
                for ax, lbs, prbs, title in [
                    (axes[0], labels_s, proba_s, "Succès commercial"),
                    (axes[1], labels_p, proba_p, "Rentabilité ROI"),
                ]:
                    s = pd.Series(prbs, index=lbs).sort_values()
                    cols_bar = [color_map.get(c, SAFFRON) for c in s.index]
                    ax.barh(s.index, s.values, color=cols_bar, alpha=0.9, height=0.5)
                    ax.set_xlim(0, 1.2)
                    ax.set_title(title, fontsize=10, fontweight="bold", color=IVORY, pad=8)
                    for i, (cls, val) in enumerate(s.items()):
                        ax.text(val + 0.04, i, f"{val:.0%}", va="center",
                                fontsize=9, fontweight="bold", color=IVORY)
                    style_axes_bw(ax, fig)
                    ax.tick_params(labelsize=9)
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                with st.expander("🎞️  Films comparables dans la base"):
                    # Filtrage sur genre + budget (±50%) + nombre d'écrans (±40%)
                    sim = data[
                        (data["Genre"] == genre_sel) &
                        (data["Budget(INR)"].between(budget_m*1e6*0.5, budget_m*1e6*1.5)) &
                        (data["Number of Screens"].between(screens*0.6, screens*1.4))
                    ].copy()
                    if sim.empty:
                        # Si trop restrictif, relâcher le filtre écrans
                        sim = data[
                            (data["Genre"] == genre_sel) &
                            (data["Budget(INR)"].between(budget_m*1e6*0.5, budget_m*1e6*1.5))
                        ].copy()
                        st.caption("ℹ️ Filtre écrans relâché — aucun film trouvé avec les deux critères.")
                    sim["Budget"]  = sim["Budget(INR)"].apply(format_inr)
                    sim["Revenue"] = sim["Revenue(INR)"].apply(format_inr)
                    sim["ROI"]     = sim["ROI"].apply(lambda x: f"{x:.1%}" if not pd.isna(x) else "N/A")
                    sim["Note"]    = sim["Rating(10)"].apply(lambda x: f"{x:.1f}/10" if not pd.isna(x) else "N/A")
                    sim["Votes"]   = sim["Votes"].apply(lambda x: f"{int(x):,}" if not pd.isna(x) else "N/A")
                    cols_show = ["Movie Name", "Budget", "Number of Screens", "Note", "Votes",
                                 "Revenue", "ROI", "Success_Class", "Profitability_Class"]
                    cols_show = [c for c in cols_show if c in sim.columns]
                    st.dataframe(sim[cols_show], use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

        else:
            st.markdown(f"""
            <div class="empty-state">
              <div class="empty-icon">🎬</div>
              <div class="empty-text">En attente d'analyse</div>
              <div class="empty-hint">
                Renseignez le genre, le budget<br>et le nombre d'écrans,<br>
                puis lancez l'analyse.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ONGLET 2 — PERFORMANCE MODÈLES
# ══════════════════════════════════════════════════════════
with tab_models:
    st.markdown(f"""
    <div class="section-ornament"><span class="section-tag">Évaluation des algorithmes</span></div>
    <p style="color:{MUTED_C};font-size:0.85rem;margin-bottom:1.5rem">
        Métriques calculées sur 20% du dataset (holdout). Features :
        <code style="background:#2A0A00;padding:2px 7px;border-radius:4px;font-size:0.8rem">
            {" &middot; ".join(FEATURES)}
        </code>
    </p>
    """, unsafe_allow_html=True)

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        model_choice = st.selectbox("Modèle", list(metrics.keys()), label_visibility="collapsed")

    m = metrics[model_choice]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy — Succès", f"{m['success_accuracy']:.1%}")
    c2.metric("F1 macro — Succès", f"{m['success_f1']:.1%}")
    c3.metric("Accuracy — ROI",    f"{m['profit_accuracy']:.1%}")
    c4.metric("F1 macro — ROI",    f"{m['profit_f1']:.1%}")

    st.markdown("<hr>", unsafe_allow_html=True)

    def plot_metrics_bar(metrics, nm, task):
        prefix = "success" if task == "success" else "profit"
        vals   = [metrics[nm][f"{prefix}_accuracy"], metrics[nm][f"{prefix}_precision"],
                  metrics[nm][f"{prefix}_recall"],   metrics[nm][f"{prefix}_f1"]]
        labels = ["Accuracy", "Precision", "Recall", "F1"]
        colors = [SAFFRON, MARIGOLD, GOLD, CRIMSON]
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="#1E0800")
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score", fontsize=9)
        tl = "Succès Commercial" if task == "success" else "Rentabilité (ROI)"
        ax.set_title(f"{nm} — {tl}", fontweight="bold", fontsize=10, color=IVORY)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9,
                    fontweight="bold", color=IVORY)
        style_axes_bw(ax, fig)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        return fig

    def plot_cm(metrics, nm, task):
        cm     = metrics[nm][f"{task}_cm"]
        labels = metrics[nm][f"{task}_labels"]
        n      = len(labels)
        # Taille augmentée pour lisibilité
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        fig.patch.set_facecolor("#1E0800")
        ax.set_facecolor("#1E0800")

        im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=MUTED_C, labelsize=9)
        cbar.ax.yaxis.set_tick_params(color=MUTED_C)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11, color=IVORY)
        ax.set_yticklabels(labels, fontsize=11, color=IVORY)
        ax.set_xlabel("Prédit", fontsize=11, color=MUTED_C, labelpad=8)
        ax.set_ylabel("Réel",   fontsize=11, color=MUTED_C, labelpad=8)
        tl = "Succès Commercial" if task == "success" else "Rentabilité (ROI)"
        ax.set_title(f"Matrice de confusion — {tl}", fontsize=12, fontweight="bold",
                     color=IVORY, pad=12)

        # Valeur centrée dans chaque cellule — couleur adaptée au fond
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                val      = cm[i, j]
                # Texte blanc sur cellule foncée, noir sur cellule claire
                txt_color = IVORY if (val / vmax) > 0.5 else "#1A0500"
                ax.text(j, i, str(val), ha="center", va="center",
                        color=txt_color, fontsize=14, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        fig.tight_layout(pad=1.5)
        return fig

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(f"<div class='section-ornament'><span class='section-tag'>Succès commercial</span></div>", unsafe_allow_html=True)
        st.pyplot(plot_metrics_bar(metrics, model_choice, "success"), use_container_width=True)
        plt.close()
        st.pyplot(plot_cm(metrics, model_choice, "success"), use_container_width=True)
        plt.close()

    with col_g2:
        st.markdown(f"<div class='section-ornament'><span class='section-tag'>Rentabilité (ROI)</span></div>", unsafe_allow_html=True)
        st.pyplot(plot_metrics_bar(metrics, model_choice, "profit"), use_container_width=True)
        plt.close()
        st.pyplot(plot_cm(metrics, model_choice, "profit"), use_container_width=True)
        plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(f"<div class='section-ornament'><span class='section-tag'>Comparaison des algorithmes</span></div>", unsafe_allow_html=True)
    rows = []
    for nm, mv in metrics.items():
        rows.append({
            "Algorithme":  nm,
            "Acc. Succès": f"{mv['success_accuracy']:.1%}",
            "F1 Succès":   f"{mv['success_f1']:.1%}",
            "Acc. ROI":    f"{mv['profit_accuracy']:.1%}",
            "F1 ROI":      f"{mv['profit_f1']:.1%}",
            "Recommandé":  "✦" if nm == best_model else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if "Random Forest" in fitted_models:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-ornament'><span class='section-tag'>Importance des variables (Random Forest)</span></div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, task, title in [
            (axes[0], "success", "Succès Commercial"),
            (axes[1], "profit",  "Rentabilité"),
        ]:
            clf = fitted_models["Random Forest"][task]
            step_key = "model" if "model" in clf.named_steps else "m"
            clf_obj = clf.named_steps[step_key] if hasattr(clf, "named_steps") else clf
            if hasattr(clf_obj, "feature_importances_"):
                imp = pd.Series(clf_obj.feature_importances_, index=FEATURES).sort_values()
                bar_colors = [GOLD if v == imp.max() else TURMERIC for v in imp.values]
                imp.plot(kind="barh", ax=ax, color=bar_colors, edgecolor="#1E0800")
                ax.set_title(f"Importances — {title}", fontsize=10, fontweight="bold", color=IVORY)
                ax.set_xlabel("Importance relative", fontsize=9)
                style_axes_bw(ax, fig)
                ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div style="margin-top:4rem;padding:1.5rem 0 1rem;
            border-top:1px solid {BORDER_C};
            text-align:center;color:{MUTED_C};font-size:0.75rem;letter-spacing:0.1em">
    ✦ &nbsp; BollywoodAI — Plateforme de conseil investissement cinéma &nbsp; ✦
    <br><span style="font-size:0.7rem;opacity:0.6;margin-top:4px;display:block">
        Random Forest &amp; Gaussian Naive Bayes &nbsp;·&nbsp; {n_films} films référencés
    </span>
</div>
""", unsafe_allow_html=True)
