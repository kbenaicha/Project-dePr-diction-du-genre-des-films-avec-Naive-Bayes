from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

st.set_page_config(page_title="Bollywood - Succès & Rentabilité", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Bollywood_Movies_data" / "bollywood_merged_clean.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    data = df[['Movie Name', 'Genre', 'Budget(INR)', 'Number of Screens', 'Rating(10)', 'Votes', 'Revenue(INR)']].copy()
    data['ROI'] = data['Revenue(INR)'] / data['Budget(INR)']
    data['Success_Class'] = pd.qcut(data['Revenue(INR)'], q=3, labels=['Flop', 'Average', 'Hit'])
    data['Profitability_Class'] = pd.qcut(data['ROI'], q=3, labels=['Low ROI', 'Medium ROI', 'High ROI'])
    return data

@st.cache_resource
def train_models():
    data = load_data().copy()
    le = LabelEncoder()
    data['Genre_Encoded'] = le.fit_transform(data['Genre'])
    features = ['Budget(INR)', 'Number of Screens', 'Rating(10)', 'Votes', 'Genre_Encoded']
    X = data[features]
    y_success = data['Success_Class']
    y_profit = data['Profitability_Class']

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_success, test_size=0.2, random_state=42, stratify=y_success
    )
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X, y_profit, test_size=0.2, random_state=42, stratify=y_profit
    )

    models = {
        'GaussianNB': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', GaussianNB())
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight='balanced'
            ))
        ])
    }

    metrics = {}
    fitted = {}

    for name, model in models.items():
        # Success
        model_s = model
        model_s.fit(X_train_s, y_train_s)
        y_pred_s = model_s.predict(X_test_s)
        ps, rs, f1s, _ = precision_recall_fscore_support(y_test_s, y_pred_s, average='macro')

        # Profitability
        model_p = Pipeline(model.steps)
        model_p.fit(X_train_p, y_train_p)
        y_pred_p = model_p.predict(X_test_p)
        pp, rp, f1p, _ = precision_recall_fscore_support(y_test_p, y_pred_p, average='macro')

        metrics[name] = {
            'success_accuracy': accuracy_score(y_test_s, y_pred_s),
            'success_precision': ps,
            'success_recall': rs,
            'success_f1': f1s,
            'profit_accuracy': accuracy_score(y_test_p, y_pred_p),
            'profit_precision': pp,
            'profit_recall': rp,
            'profit_f1': f1p,
            'success_cm': confusion_matrix(y_test_s, y_pred_s, labels=sorted(y_success.unique())),
            'profit_cm': confusion_matrix(y_test_p, y_pred_p, labels=sorted(y_profit.unique())),
            'success_labels': sorted(y_success.unique()),
            'profit_labels': sorted(y_profit.unique()),
        }

        fitted[name] = {
            'success': model_s.fit(X, y_success),
            'profit': model_p.fit(X, y_profit)
        }

    return data, le, features, metrics, fitted

data, genre_encoder, features, metrics, fitted_models = train_models()

st.title("Analyse du succès commercial et de la rentabilité des films Bollywood")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Présentation", "Exploration", "Succès", "Rentabilité", "Prédiction"
])

with tab1:
    st.header("Contexte du projet")
    st.write(
        "Cette application analyse les films Bollywood à partir de cinq variables : "
        "**Budget(INR)**, **Number of Screens**, **Rating(10)**, **Votes** et **Genre**."
    )
    st.write(
        "Deux objectifs sont étudiés : "
        "la **classe de succès commercial** et la **classe de rentabilité**."
    )
    st.write(
        "Deux modèles sont comparés : **Gaussian Naive Bayes** et **Random Forest**."
    )
    st.subheader("Variables utilisées")
    st.write(data[['Genre', 'Budget(INR)', 'Number of Screens', 'Rating(10)', 'Votes', 'Revenue(INR)', 'ROI']].head())

with tab2:
    st.header("Exploration des données")
    st.dataframe(data.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        data['Revenue(INR)'].hist(bins=30, ax=ax)
        ax.set_title("Distribution du revenu")
        ax.set_xlabel("Revenue(INR)")
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        data['ROI'].hist(bins=30, ax=ax)
        ax.set_title("Distribution du ROI")
        ax.set_xlabel("ROI")
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)

    st.subheader("Analyse par genre")
    genre_stats = data.groupby('Genre')[['Revenue(INR)', 'ROI', 'Votes', 'Rating(10)', 'Budget(INR)', 'Number of Screens']].mean().sort_values('Revenue(INR)', ascending=False)
    st.dataframe(genre_stats, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    genre_stats['Revenue(INR)'].plot(kind='bar', ax=ax)
    ax.set_title("Revenu moyen par genre")
    ax.set_ylabel("Revenue(INR)")
    st.pyplot(fig)

with tab3:
    st.header("Résultats — Succès commercial")
    rows = []
    for model_name, vals in metrics.items():
        rows.append({
            'Modèle': model_name,
            'Accuracy': vals['success_accuracy'],
            'Precision macro': vals['success_precision'],
            'Recall macro': vals['success_recall'],
            'F1 macro': vals['success_f1']
        })
    st.dataframe(pd.DataFrame(rows).sort_values('F1 macro', ascending=False), use_container_width=True)

    selected_model = st.selectbox("Choisir un modèle pour voir la matrice de confusion (succès)", list(metrics.keys()), key='succ')
    cm = metrics[selected_model]['success_cm']
    labels = metrics[selected_model]['success_labels']
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    fig.colorbar(im)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"Matrice de confusion - {selected_model}")
    st.pyplot(fig)

with tab4:
    st.header("Résultats — Rentabilité")
    rows = []
    for model_name, vals in metrics.items():
        rows.append({
            'Modèle': model_name,
            'Accuracy': vals['profit_accuracy'],
            'Precision macro': vals['profit_precision'],
            'Recall macro': vals['profit_recall'],
            'F1 macro': vals['profit_f1']
        })
    st.dataframe(pd.DataFrame(rows).sort_values('F1 macro', ascending=False), use_container_width=True)

    selected_model = st.selectbox("Choisir un modèle pour voir la matrice de confusion (rentabilité)", list(metrics.keys()), key='prof')
    cm = metrics[selected_model]['profit_cm']
    labels = metrics[selected_model]['profit_labels']
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    fig.colorbar(im)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"Matrice de confusion - {selected_model}")
    st.pyplot(fig)

with tab5:
    st.header("Prédiction interactive")
    model_choice = st.selectbox("Modèle", list(fitted_models.keys()))
    genre = st.selectbox("Genre", sorted(data['Genre'].unique()))
    budget = st.number_input("Budget(INR)", min_value=1.0, value=float(data['Budget(INR)'].median()))
    screens = st.number_input("Number of Screens", min_value=1.0, value=float(data['Number of Screens'].median()))
    rating = st.slider("Rating(10)", min_value=0.0, max_value=10.0, value=float(round(data['Rating(10)'].median(), 1)), step=0.1)
    votes = st.number_input("Votes", min_value=0.0, value=float(data['Votes'].median()))

    if st.button("Lancer la prédiction"):
        genre_encoded = genre_encoder.transform([genre])[0]
        sample = pd.DataFrame([{
            'Budget(INR)': budget,
            'Number of Screens': screens,
            'Rating(10)': rating,
            'Votes': votes,
            'Genre_Encoded': genre_encoded
        }])

        success_model = fitted_models[model_choice]['success']
        profit_model = fitted_models[model_choice]['profit']

        success_pred = success_model.predict(sample)[0]
        success_proba = success_model.predict_proba(sample)[0]
        profit_pred = profit_model.predict(sample)[0]
        profit_proba = profit_model.predict_proba(sample)[0]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Succès commercial")
            st.metric("Classe prédite", success_pred)
            st.write(pd.DataFrame({
                'Classe': success_model.classes_,
                'Probabilité': success_proba
            }))
        with c2:
            st.subheader("Rentabilité")
            st.metric("Classe prédite", profit_pred)
            st.write(pd.DataFrame({
                'Classe': profit_model.classes_,
                'Probabilité': profit_proba
            }))
