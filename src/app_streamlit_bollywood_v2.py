from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Bollywood - Succès & Rentabilité",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Bollywood_Movies_data" / "bollywood_merged_clean.csv"
ACCENT = "#c44900"
DARK = "#1f1a17"
SAND = "#f6efe8"


st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(196, 73, 0, 0.12), transparent 28%),
            linear-gradient(180deg, #fbf7f2 0%, #fffdf9 100%);
        color: {DARK};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .hero-card {{
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #2b211c 0%, #58321f 55%, #c44900 100%);
        color: #fff9f3;
        box-shadow: 0 18px 50px rgba(84, 51, 27, 0.22);
        margin-bottom: 1rem;
    }}
    .hero-kicker {{
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-size: 0.78rem;
        opacity: 0.85;
        margin-bottom: 0.4rem;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.6rem;
    }}
    .hero-copy {{
        font-size: 1rem;
        max-width: 760px;
        opacity: 0.95;
    }}
    .mini-card {{
        background: rgba(255, 249, 243, 0.88);
        border: 1px solid rgba(196, 73, 0, 0.14);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 25px rgba(115, 76, 45, 0.08);
        min-height: 126px;
    }}
    .mini-label {{
        font-size: 0.84rem;
        color: #7f5a46;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }}
    .mini-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: {DARK};
        margin-bottom: 0.2rem;
    }}
    .section-card {{
        background: rgba(255, 253, 249, 0.92);
        border: 1px solid rgba(92, 56, 31, 0.1);
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        margin-bottom: 1rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 999px;
        padding: 0.55rem 1rem;
        background: rgba(196, 73, 0, 0.08);
    }}
    .stTabs [aria-selected="true"] {{
        background: {ACCENT};
        color: white;
    }}
    [data-testid="stMetric"] {{
        background: rgba(255, 250, 245, 0.95);
        border: 1px solid rgba(196, 73, 0, 0.12);
        padding: 1rem;
        border-radius: 18px;
    }}
    .sidebar-note {{
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: {SAND};
        border: 1px solid rgba(196, 73, 0, 0.15);
        font-size: 0.92rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def format_inr(value):
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} B INR"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M INR"
    if value >= 1_000:
        return f"{value / 1_000:.1f} K INR"
    return f"{value:.0f} INR"


def style_axes(ax):
    ax.set_facecolor("#fffaf5")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.tick_params(colors="#50382b")
    ax.title.set_color(DARK)
    ax.xaxis.label.set_color(DARK)
    ax.yaxis.label.set_color(DARK)


def plot_histogram(series, title, xlabel, color):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series.dropna(), bins=30, color=color, edgecolor="white", alpha=0.92)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequence")
    style_axes(ax)
    fig.tight_layout()
    return fig


def plot_bar(values, title, ylabel, color):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    values.plot(kind="bar", ax=ax, color=color, width=0.72)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    style_axes(ax)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = ax.imshow(cm, cmap="Oranges")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color=DARK, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_probability_chart(classes, probabilities, title, color):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    probs = pd.Series(probabilities, index=classes).sort_values()
    ax.barh(probs.index, probs.values, color=color, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(title, fontsize=12, fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    return fig


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    data = df[
        [
            "Movie Name",
            "Genre",
            "Budget(INR)",
            "Number of Screens",
            "Rating(10)",
            "Votes",
            "Revenue(INR)",
        ]
    ].copy()
    data["ROI"] = data["Revenue(INR)"] / data["Budget(INR)"]
    data["Success_Class"] = pd.qcut(data["Revenue(INR)"], q=3, labels=["Flop", "Average", "Hit"])
    data["Profitability_Class"] = pd.qcut(
        data["ROI"], q=3, labels=["Low ROI", "Medium ROI", "High ROI"]
    )
    return data


@st.cache_resource
def train_models():
    data = load_data().copy()
    le = LabelEncoder()
    data["Genre_Encoded"] = le.fit_transform(data["Genre"])
    features = ["Budget(INR)", "Number of Screens", "Rating(10)", "Votes", "Genre_Encoded"]
    X = data[features]
    y_success = data["Success_Class"]
    y_profit = data["Profitability_Class"]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_success, test_size=0.2, random_state=42, stratify=y_success
    )
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X, y_profit, test_size=0.2, random_state=42, stratify=y_profit
    )

    models = {
        "GaussianNB": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", GaussianNB())]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }

    metrics = {}
    fitted = {}

    for name, model in models.items():
        model_s = model
        model_s.fit(X_train_s, y_train_s)
        y_pred_s = model_s.predict(X_test_s)
        ps, rs, f1s, _ = precision_recall_fscore_support(
            y_test_s, y_pred_s, average="macro"
        )

        model_p = Pipeline(model.steps)
        model_p.fit(X_train_p, y_train_p)
        y_pred_p = model_p.predict(X_test_p)
        pp, rp, f1p, _ = precision_recall_fscore_support(
            y_test_p, y_pred_p, average="macro"
        )

        metrics[name] = {
            "success_accuracy": accuracy_score(y_test_s, y_pred_s),
            "success_precision": ps,
            "success_recall": rs,
            "success_f1": f1s,
            "profit_accuracy": accuracy_score(y_test_p, y_pred_p),
            "profit_precision": pp,
            "profit_recall": rp,
            "profit_f1": f1p,
            "success_cm": confusion_matrix(y_test_s, y_pred_s, labels=sorted(y_success.unique())),
            "profit_cm": confusion_matrix(y_test_p, y_pred_p, labels=sorted(y_profit.unique())),
            "success_labels": sorted(y_success.unique()),
            "profit_labels": sorted(y_profit.unique()),
        }

        fitted[name] = {
            "success": model_s.fit(X, y_success),
            "profit": model_p.fit(X, y_profit),
        }

    return data, le, features, metrics, fitted


data, genre_encoder, features, metrics, fitted_models = train_models()
best_success_model = max(metrics, key=lambda name: metrics[name]["success_f1"])
best_profit_model = max(metrics, key=lambda name: metrics[name]["profit_f1"])
top_genre = data.groupby("Genre")["Revenue(INR)"].mean().idxmax()


with st.sidebar:
    st.markdown("## Tableau de bord")
    st.markdown(
        """
        <div class="sidebar-note">
        Cette application compare deux modeles de classification pour estimer
        le succes commercial et la rentabilite des films Bollywood.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Points cles")
    st.write(f"Jeu de donnees: {len(data)} films")
    st.write(f"Meilleur modele succes: {best_success_model}")
    st.write(f"Meilleur modele rentabilite: {best_profit_model}")
    st.write(f"Genre le plus performant: {top_genre}")


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Bollywood Analytics Lab</div>
        <div class="hero-title">Predire le succes commercial et la rentabilite d'un film</div>
        <div class="hero-copy">
            Une interface de demonstration pour explorer les donnees, comparer les performances
            de modeles supervises et lancer une prediction interactive a partir des variables clefs.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">Films analyses</div>
            <div class="mini-value">{len(data)}</div>
            <div>Base nettoyee exploitee pour les deux modeles.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">Revenu median</div>
            <div class="mini-value">{format_inr(data["Revenue(INR)"].median())}</div>
            <div>Niveau central de performance commerciale observe.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">ROI median</div>
            <div class="mini-value">{data["ROI"].median():.2f}x</div>
            <div>Rapport revenu sur budget au centre de l'echantillon.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">Genre dominant</div>
            <div class="mini-value">{top_genre}</div>
            <div>Genre avec le revenu moyen le plus eleve.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Vue d'ensemble", "Exploration", "Succes", "Rentabilite", "Prediction"]
)

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    intro_col, metrics_col = st.columns([1.2, 1])
    with intro_col:
        st.subheader("Contexte analytique")
        st.write(
            "Le projet etudie la performance des films Bollywood a partir de cinq variables "
            "explicatives: budget, nombre d'ecrans, note, votes et genre."
        )
        st.write(
            "Deux cibles sont modelisees: la classe de succes commercial et la classe de "
            "rentabilite. L'application compare un modele probabiliste simple et un ensemble "
            "plus robuste pour montrer les ecarts de comportement."
        )
    with metrics_col:
        st.metric("Meilleur F1 - Succes", f'{metrics[best_success_model]["success_f1"]:.3f}')
        st.metric("Meilleur F1 - Rentabilite", f'{metrics[best_profit_model]["profit_f1"]:.3f}')
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Variables utilisees")
    st.dataframe(
        data[
            [
                "Movie Name",
                "Genre",
                "Budget(INR)",
                "Number of Screens",
                "Rating(10)",
                "Votes",
                "Revenue(INR)",
                "ROI",
            ]
        ].head(12),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Apercu des donnees")
    st.dataframe(data.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(
            plot_histogram(
                data["Revenue(INR)"], "Distribution du revenu", "Revenue (INR)", ACCENT
            )
        )
    with chart_col2:
        st.pyplot(plot_histogram(data["ROI"], "Distribution du ROI", "ROI", "#d98f3d"))

    genre_stats = (
        data.groupby("Genre")[
            ["Revenue(INR)", "ROI", "Votes", "Rating(10)", "Budget(INR)", "Number of Screens"]
        ]
        .mean()
        .sort_values("Revenue(INR)", ascending=False)
    )

    stats_col1, stats_col2 = st.columns([1.15, 1])
    with stats_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Performance moyenne par genre")
        st.dataframe(genre_stats.round(2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with stats_col2:
        st.pyplot(
            plot_bar(
                genre_stats["Revenue(INR)"].head(8),
                "Top genres par revenu moyen",
                "Revenue moyen (INR)",
                "#99582a",
            )
        )

with tab3:
    success_rows = [
        {
            "Modele": model_name,
            "Accuracy": vals["success_accuracy"],
            "Precision macro": vals["success_precision"],
            "Recall macro": vals["success_recall"],
            "F1 macro": vals["success_f1"],
        }
        for model_name, vals in metrics.items()
    ]
    success_df = pd.DataFrame(success_rows).sort_values("F1 macro", ascending=False)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Comparaison des modeles pour le succes commercial")
    st.dataframe(success_df.style.format({"Accuracy": "{:.3f}", "Precision macro": "{:.3f}", "Recall macro": "{:.3f}", "F1 macro": "{:.3f}"}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    model_col1, model_col2 = st.columns([0.8, 1.2])
    with model_col1:
        selected_model = st.selectbox(
            "Choisir un modele pour la matrice de confusion",
            list(metrics.keys()),
            key="succ",
        )
        st.metric("Accuracy", f'{metrics[selected_model]["success_accuracy"]:.3f}')
        st.metric("F1 macro", f'{metrics[selected_model]["success_f1"]:.3f}')
    with model_col2:
        st.pyplot(
            plot_confusion_matrix(
                metrics[selected_model]["success_cm"],
                metrics[selected_model]["success_labels"],
                f"Matrice de confusion - {selected_model}",
            )
        )

with tab4:
    profit_rows = [
        {
            "Modele": model_name,
            "Accuracy": vals["profit_accuracy"],
            "Precision macro": vals["profit_precision"],
            "Recall macro": vals["profit_recall"],
            "F1 macro": vals["profit_f1"],
        }
        for model_name, vals in metrics.items()
    ]
    profit_df = pd.DataFrame(profit_rows).sort_values("F1 macro", ascending=False)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Comparaison des modeles pour la rentabilite")
    st.dataframe(profit_df.style.format({"Accuracy": "{:.3f}", "Precision macro": "{:.3f}", "Recall macro": "{:.3f}", "F1 macro": "{:.3f}"}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    model_col1, model_col2 = st.columns([0.8, 1.2])
    with model_col1:
        selected_model = st.selectbox(
            "Choisir un modele pour la matrice de confusion",
            list(metrics.keys()),
            key="profit",
        )
        st.metric("Accuracy", f'{metrics[selected_model]["profit_accuracy"]:.3f}')
        st.metric("F1 macro", f'{metrics[selected_model]["profit_f1"]:.3f}')
    with model_col2:
        st.pyplot(
            plot_confusion_matrix(
                metrics[selected_model]["profit_cm"],
                metrics[selected_model]["profit_labels"],
                f"Matrice de confusion - {selected_model}",
            )
        )

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Simulation d'un nouveau film")
    form_col, output_col = st.columns([0.95, 1.05])

    with form_col:
        model_choice = st.selectbox("Modele", list(fitted_models.keys()))
        genre = st.selectbox("Genre", sorted(data["Genre"].unique()))
        budget = st.number_input(
            "Budget (INR)",
            min_value=1.0,
            value=float(data["Budget(INR)"].median()),
            step=100000.0,
        )
        screens = st.number_input(
            "Nombre d'ecrans",
            min_value=1.0,
            value=float(data["Number of Screens"].median()),
            step=10.0,
        )
        rating = st.slider(
            "Note (/10)",
            min_value=0.0,
            max_value=10.0,
            value=float(round(data["Rating(10)"].median(), 1)),
            step=0.1,
        )
        votes = st.number_input(
            "Votes",
            min_value=0.0,
            value=float(data["Votes"].median()),
            step=100.0,
        )
        launch_prediction = st.button("Lancer la prediction", use_container_width=True)

    with output_col:
        st.info(
            "Le modele renvoie une classe predite pour le succes commercial et pour la "
            "rentabilite, avec les probabilites associees."
        )

        if launch_prediction:
            genre_encoded = genre_encoder.transform([genre])[0]
            sample = pd.DataFrame(
                [
                    {
                        "Budget(INR)": budget,
                        "Number of Screens": screens,
                        "Rating(10)": rating,
                        "Votes": votes,
                        "Genre_Encoded": genre_encoded,
                    }
                ]
            )

            success_model = fitted_models[model_choice]["success"]
            profit_model = fitted_models[model_choice]["profit"]

            success_pred = success_model.predict(sample)[0]
            success_proba = success_model.predict_proba(sample)[0]
            profit_pred = profit_model.predict(sample)[0]
            profit_proba = profit_model.predict_proba(sample)[0]

            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                st.metric("Succes commercial predit", success_pred)
                st.pyplot(
                    plot_probability_chart(
                        success_model.classes_,
                        success_proba,
                        "Probabilites - Succes",
                        ACCENT,
                    )
                )
            with pred_col2:
                st.metric("Rentabilite predite", profit_pred)
                st.pyplot(
                    plot_probability_chart(
                        profit_model.classes_,
                        profit_proba,
                        "Probabilites - Rentabilite",
                        "#d98f3d",
                    )
                )
        else:
            st.markdown(
                """
                <div class="mini-card">
                    Renseigne les caracteristiques du film puis lance la prediction pour afficher
                    les classes estimees et les probabilites de chaque scenario.
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
