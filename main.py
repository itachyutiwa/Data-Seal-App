# Core Pkgs
import scipy
from sklearn.decomposition import PCA
import numpy as np
import streamlit as st
import altair as alt
# EDA Pkgs
import pandas as pd

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import stats, pearsonr, fisher_exact
from scipy.stats import shapiro
from mesFonctions import *
from scipy.stats import f

matplotlib.use("Agg")
matplotlib.use('TkAgg')
import seaborn as sns


def main():
    """Semi Automated ML App with Streamlit """

    activities = ["ANALYSE EXPLORATOIRE", "LES TESTS STATISTIQUES", "ACP", "ANALYSE FACTORIELLE DES COMPOSANTES",
                  "MACHINE LEARNING",
                  "PREVISION", "QUELQUES GRAPHIQUES"]
    choice = st.sidebar.selectbox("Choisissez une action", activities)
    les_types = ["Quantitatives", "Qualitatives"]
    st.title("DATA - SEAL - APP")
    data = chargerDonnees()
    if choice == 'ANALYSE EXPLORATOIRE':
        if data is not None:

            if data is not None:

                try:
                    df = pd.read_csv(data)
                    var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
                    # var_quali = df.select_dtypes(exclude=['float64', 'float32', 'double', 'int32', 'int64', 'date'])
                    all_columns = df.columns.to_list()
                except:
                    pass

                try:
                    if st.checkbox("Choix de l'année"):
                        annee = [2017, 2018, 2019, 2020, 'Toutes les années']
                        choix = st.selectbox('Choisissez une option ', annee)
                        if choix != 'Toutes les années':
                            df = df[df['annee'] == choix]
                        else:
                            df = pd.read_csv(data)

                        all_columns = df.columns.to_list()
                except:
                    pass

                try:
                    df = pd.read_table(data)
                    all_columns = df.columns.to_list()
                except:
                    pass
                st.subheader('Informations générales sur la base de données')
                if st.checkbox("Aperçu de quelques lignes de notre base de données"):
                    apercuDonnees(df)

                if st.checkbox("Dimensions des données"):
                    dimensionDonnees(df)
                if st.checkbox("Valeur manquantes"):
                    Nan(df)

                if st.checkbox("Valeur aberrante"):
                    Aberrantes(df)

                # if st.checkbox("Pourcentage valeurs manquantes"):
                #     column_name = df.columns.to_list()
                #     NaN_percent(df, column_name)
                # if st.checkbox("Information sur la base de données"):
                #     infoDonnees(df)

                if st.checkbox("Descriptions de nos données"):
                    descriptionDonnees(df)
                if st.checkbox("Afficher les colunnes"):
                    colonnesDonnees(df)

                if st.checkbox("Afficher les colonnes selectionnées"):
                    colonnesSelectionnees(df)

                # ------------------------------------------------------------------------------------------------------------------------------------------------
                # -----------------------------------------------------QUANTITATIVES-----------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                # Stattistiques à faire->Description,extreme, etendu,intervalle interquartile,coefficient de variation, mode
                st.subheader("Analyse univariée et multivariées")
                type_variables = st.selectbox('Précisez le type de variable ', les_types)
                if type_variables == "Quantitatives":

                    if st.checkbox("Etendu"):
                        etenduQuanti(df)

                    if st.checkbox("Variance"):
                        varianceQuanti(df)

                    if st.checkbox("Coefficient de variation"):
                        coeffVariationQuanti(df)

                    if st.checkbox(" Intervalle interquartile"):
                        intervQuanti(df)

                    if st.checkbox("Rapport de variation"):
                        rapportVariationQuanti(df)
                    if st.checkbox('Nuage de points moyenne dossier et moyenne definitive'):
                        pass

                # ------------------------------------FIN QUANTITATIFS-----------------------------------------------------------------------------------------

                # ---------------------------------------------------------------------------------------------------------------------------------------------
                # -----------------------------------------------------QUALITATIVES---------------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------------------------------------------------------

                if type_variables == "Qualitatives":
                    all_columns = df.columns.to_list()
                    if 'genre' in all_columns:
                        if st.checkbox("Afficher les effectifs par genre"):
                            effectifGenre(df)

                    if 'nationalite' in all_columns:
                        if st.checkbox("Afficher les effectifs par nationalité"):
                            effectifNationalite(df)

                    if 'mentionAuBac' in all_columns:
                        if st.checkbox("Afficher les effectifs par mention de bac"):
                            effectifMention(df)
                    if 'redoublementTerminale' in all_columns or 'etabs' in all_columns:
                        if st.checkbox("Afficher les effectifs par redoublement terminale"):
                            effectifRedoub(df)

                    if 'matiere' in all_columns:
                        if st.checkbox("Afficher les effectifs par matiere"):
                            effectifMatiere(df)

                    if 'rangTerminale' in all_columns:
                        if st.checkbox("Afficher les effectifs par rang terminale"):
                            effectifRangTerminale(df)

                    if 'serie' in all_columns:
                        if st.checkbox("Afficher les effectifs par serie de bac"):
                            effectifSerie(df)

                    if 'filiereAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs par filière d'accueil"):
                            effectifFiliereAcc(df)

                    if 'cycleAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs par cycle d'accueil"):
                            effectifCycleAcc(df)

                    if 'ecoleAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs par école d'accueil"):
                            ecoleAcc(df)

                    if 'cycles' in all_columns:
                        if st.checkbox("Afficher les effectifs par cycle de choix"):
                            effectifcycles(df)

                    if 'codeFiliere' in all_columns:
                        if st.checkbox("Afficher les effectifs par filière de choix"):
                            effectifcodeFilire(df)

                    if 'codeFiliere' in all_columns and 'genre' in all_columns:
                        if st.checkbox("Afficher les effectifs par filière de choix selon le genre"):
                            effectifcodeFilireGenre(df)

                    if 'etablissementOrigine' in all_columns or 'etabs' in all_columns:
                        if st.checkbox("Afficher les effectifs des candidats par établissement d'origine"):
                            effectifEtabOrigine(df)

                    if 'drens' in all_columns:
                        if st.checkbox("Afficher les effectifs des candidats par dren"):
                            effectifDren(df)

                    # ------------------------------------------------------
                    if 'serie' in all_columns or 'genre' in all_columns:
                        if st.checkbox("Afficher les effectifs par serie de bac selon le genre"):
                            effectifSerieGenre(df)

                    if 'genre' in all_columns and 'filiereAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs des candidat par filiere selon le genre"):
                            effectifFilereGenre(df)

                    if 'genre' in all_columns and 'codeEcole' in all_columns:
                        if st.checkbox("Afficher les effectifs des candidat par ecole accueille selon le genre"):
                            effectifEcoleGenre(df)

                    if 'drens' in all_columns and 'filiereAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs des candidat par filiere selon le dren"):
                            effectifFilereDren(df)

                    if 'drens' in all_columns and 'ecoleAccueil' in all_columns and 'cycleAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs par ecole accueille selon le dren"):
                            effectifEcoleAccDren(df)
                    # ------------------

                    if 'rangTerminale' in all_columns or 'drens' in all_columns:
                        if st.checkbox("Afficher les effectifs par rang terminale selon le dren"):
                            effectifRangTerminaleDRen(df)

                    if 'serie' in all_columns or 'drens' in all_columns:
                        if st.checkbox("Afficher les effectifs par serie de bac selon le dren"):
                            effectifSerieDRen(df)

                    if 'serie' in all_columns and 'filiereAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs par filière selon la serie"):
                            effectifFiliereSerie(df)

                    # -------------------------------------------------
                    # ------------------
                    if 'etablissementOrigine' in all_columns or 'etabs' in all_columns and 'genre' in all_columns:
                        if st.checkbox(
                                "Afficher les effectifs des candidats par établissement d'origine selon le genre"):
                            effectifEtabOrigineGenre(df)

                    if 'drens' in all_columns or 'ecoleAccueil' in all_columns or 'cycleAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs par dren selon le genre"):
                            effectifDrenGenre(df)


                    if 'rangTerminale' in all_columns or 'genre' in all_columns:
                        if st.checkbox("Afficher les effectifs par rang terminale selon le genre"):
                            effectifRangTerminaleGenre(df)


                    if 'nationalite' in all_columns and 'genre' in all_columns:
                        if st.checkbox("Afficher les effectifs des candidats par nationalite selon le genre"):
                            effectifNationaliteGenre(df)

                    if 'serie' in all_columns and 'ecoleAccueil' in all_columns:
                        if st.checkbox("Afficher les effectifs des candidats par ecole accueille selon le genre"):
                            effectifEcoleGenre(df)
                    # -------------------------------------------------
                    if 'drens' in all_columns and 'filiereAccueil' in all_columns and 'genre' in all_columns:
                        if st.checkbox("Afficher les effectifs par dren, filire accueille selon le genre"):
                            effectifDrenFilAccGenre(df)

#-----------------------------------------------------------------------------
                    if 'filiereAccueil' in all_columns or 'serie' in all_columns or 'drens' \
                                                                                    'matiere' in all_columns or 'codeEcole' in all_columns or 'codeFiliere' in all_columns or \
                            'cycles' in all_columns or 'genre' in all_columns or 'nationalite' in all_columns or \
                            'cycleAccueil' in all_columns or 'mentionAuBac' in all_columns or 'cycleAccueil' in all_columns or \
                            'ecoleAccueil' in all_columns or 'drens' in all_columns or 'ecoleAccueil' in all_columns or \
                            'cycleAccueil' in all_columns:
                        if st.checkbox('Les histogrammes'):
                            selected_columns_names = st.multiselect("Choisir", all_columns)
                            fig, ax = plt.subplots()
                            cust_data = df[selected_columns_names]
                            ax.hist(cust_data, bins=20)
                            st.pyplot(fig)
                        if st.checkbox('Histogrammes téléchargeable'):
                            fig = plt.figure()
                            fig, ax = plt.subplots()
                            selected_columns_names = st.multiselect("Choisir", all_columns)
                            cust_data = df[selected_columns_names]
                            ax.hist(cust_data, bins=20)
                            st.plotly_chart(fig)
                            st.pyplot(fig)

                        if st.checkbox('Autres types de graphiques'):

                            all_columns_names = df.columns.to_list()
                            type_of_plot = st.selectbox("Choisissez le type de graphique",
                                                        ["line", "box", "kde", "camamber"])
                            if type_of_plot == 'camamber':
                                selected_columns_names = st.selectbox("Choisissez la colonne à representer",
                                                                      all_columns_names)
                            else:
                                selected_columns_names = st.multiselect("Choisissez la colonne à representer",
                                                                        all_columns_names)
                            if st.button("Construire le graphe"):
                                st.success(
                                    "Construction du graphique {} de {} ...".format(type_of_plot,
                                                                                    selected_columns_names))

                                # Plot By Streamlit
                                if type_of_plot == 'area':
                                    cust_data = df[selected_columns_names]
                                    st.area_chart(cust_data)

                                elif type_of_plot == 'bar':
                                    cust_data = df[selected_columns_names]
                                    st.bar_chart(cust_data)

                                elif type_of_plot == 'camamber':
                                    fig = plt.figure()
                                    cust_data = df[selected_columns_names]
                                    explode = (0, 0.15)
                                    plt.pie(cust_data.value_counts(), explode=explode, labels=cust_data.value_counts(),
                                            autopct='%1.1f%%', startangle=90, shadow=True)
                                    plt.axis('equal')
                                    plt.show()


                                elif type_of_plot:
                                    cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                                    st.write(cust_plot)
                                    st.pyplot()
                        if st.checkbox('Les tendances au fil des années'):
                            fig = plt.figure()
                            plt.grid(True)
                            # ---Postulants----Admissibles-----Valides----Admis---candidats_postulants.csv---

                            if choix == 2017:
                                m = [0, 5387, 1111, 5181, 473]
                                f = [0, 2724, 312, 2619, 111]
                                i = [0, 0, 0, 0, 0]
                                plt.plot([0, 5000, 6000, 7000, 8000], m, "b", linewidth=0.8, marker="*", label="M")
                                plt.plot([0, 5000, 6000, 7000, 8000], f, "r", linewidth=0.8, marker="+", label="F")
                                plt.plot([0, 5000, 6000, 7000, 8000], i, "o", linewidth=0.8, marker=".", label="I")
                                plt.legend()
                                plt.figure(figsize=(12.0, 12.0))
                                plt.title('Evolution selon l\n année 2017')
                                plt.suptitle('Postulants  Admissibles   Valides  Admis')
                                st.pyplot(fig)
                            elif choix == 2018:
                                m = [0, 4355, 4159, 4159, 448]
                                f = [0, 2303, 2230, 2230, 150]
                                i = [0, 0, 0, 0, 0]
                                plt.plot([0, 5000, 6000, 7000, 8000], m, "b", linewidth=0.8, marker="*", label="M")
                                plt.plot([0, 5000, 6000, 7000, 8000], f, "r", linewidth=0.8, marker="+", label="F")
                                plt.plot([0, 5000, 6000, 7000, 8000], i, "o", linewidth=0.8, marker="+", label="I")
                                plt.legend()
                                plt.title('Evolution selon l\n année 2018')
                                plt.suptitle('Postulants  Admissibles   Valides  Admis')
                                st.pyplot(fig)
                            elif choix == 2019:
                                m = [0, 4199, 1023, 4017, 531]
                                f = [0, 2403, 409, 2307, 191]
                                i = [0, 0, 0, 0, 0]
                                plt.plot([0, 10000, 20000, 30000, 40000], m, "b", linewidth=0.8, marker="*", label="M")
                                plt.plot([0, 10000, 20000, 30000, 40000], f, "r", linewidth=0.8, marker="+", label="F")
                                plt.plot([0, 10000, 20000, 30000, 40000], i, "o", linewidth=0.8, marker="+", label="I")
                                plt.legend()
                                plt.title('Evolution selon l\n année 2018')
                                plt.suptitle('Postulants  Admissibles   Valides  Admis')
                                st.pyplot(fig)
                            else:
                                m = [0, 4546, 4546, 4546, 452]
                                f = [0, 2730, 2730, 2730, 203]
                                i = [0, 1382, 0, 0, 0]
                                plt.plot([0, 10000, 20000, 30000, 40000], m, "b", linewidth=0.8, marker="*", label="M")
                                plt.plot([0, 10000, 20000, 30000, 40000], f, "r", linewidth=0.8, marker="+", label="F")
                                plt.plot([0, 10000, 20000, 30000, 40000], i, "g", linewidth=0.8, marker="+", label="I")
                                plt.legend()
                                plt.title('Evolution selon toutes les années')
                                plt.suptitle('Postulants  Admissibles   Valides  Admis')
                                st.pyplot(fig)

                        if choix == 2017:
                            if st.checkbox('Camamber - Proportion de filles et garçon'):
                                Postulants = [5387, 2724]
                                Admissibles = [1111, 312]
                                Valides = [5181, 2619]
                                Admis = [473, 111]
                                tendances = ['Postulants', 'Admissibles', 'Valides', 'Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Postulants':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Postulants, labels=['M: 66.42 %', 'F: 33.58 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Postulants 2017")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admissibles':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admissibles, labels=['M: 78.07 %', 'F 21.93 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admissibles 2017")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Valides':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Valides, labels=['M: 66.42 %', 'F 33.58 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Valides 2017")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admis, labels=['M: 80.99 %', 'F: 19.01 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admis 2017")
                                    plt.legend()
                                    st.pyplot(fig)

                        elif choix == 2018:
                            if st.checkbox('Camamber - Proportion de filles et garçon'):
                                Postulants = [4355, 2303]
                                Admissibles = [4159, 2230]
                                Valides = [4159, 2230]
                                Admis = [448, 150]
                                tendances = ['Postulants', 'Admissibles', 'Valides', 'Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Postulants':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Postulants, labels=['M: 65.41 %', 'F: 34.50 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Postulants 2018")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admissibles':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admissibles, labels=['M: 65.10 %', 'F: 34.90 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admissibles 2018")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Valides':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Valides, labels=['M: 65.10 %', 'F: 34.90 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Valides 2018")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admis, labels=['M: 74.92 %', 'F: 25.08 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admis 2018")
                                    plt.legend()

                                    st.pyplot(fig)
                        elif choix == 2019:
                            if st.checkbox('Camamber - Proportion de filles et garçon'):
                                Postulants = [4199, 2403]
                                Admissibles = [1023, 409]
                                Valides = [4017, 2307]
                                Admis = [531, 191]
                                tendances = ['Postulants', 'Admissibles', 'Valides', 'Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Postulants':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Postulants, labels=['M: 63.60 %', 'F: 36.40 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Postulants 2019")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admissibles':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admissibles, labels=['M: 71.44 %', 'F: 28.56 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admissibles 2019")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Valides':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Valides, labels=['M: 63.52 %', 'F: 36.48 %'], normalize=True)
                                    plt.title("Proportion H/F candidats Valides 2019")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admis, labels=['M: 73.55 %', 'F: 26.45 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admis 2019")
                                    plt.legend()

                                    st.pyplot(fig)

                        elif choix == 2020:
                            if st.checkbox('Camamber - Proportion de filles et garçon'):
                                Postulants = [4546, 2730, 1382]
                                Admissibles = [4546, 2730]
                                Valides = [4017, 2307]
                                Admis = [452, 203]
                                tendances = ['Postulants', 'Admissibles', 'Valides', 'Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Postulants':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Postulants, labels=['M: 52.50 %', 'F: 31.53 %', 'I: 15.96 %'],
                                            normalize=True)
                                    plt.title("Proportion M/F candidats Postulants 2020")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admissibles':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admissibles, labels=['M: 62.48 %', 'F: 37.52 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admissibles 2020")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Valides':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Valides, labels=['M: 63.52 %', 'F: 37.52 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Valides 2020")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admis, labels=['M: 69.01 %', 'F: 36.48 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admis 2020")
                                    plt.legend()

                                    st.pyplot(fig)
                        elif choix == 'Toutes les années':
                            if st.checkbox('Camamber - Proportion de filles et garçon'):
                                Postulants = [18487, 10160, 1382]
                                Admissibles = [10839, 5681]
                                Valides = [17903, 9886]
                                Admis = [1904, 655]
                                tendances = ['Postulants', 'Admissibles', 'Valides', 'Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Postulants':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Postulants, labels=['M: 61.62 %', 'F: 33.83 %', 'I: 4.60 %'],
                                            normalize=True)
                                    plt.title("Proportion M/F candidats Postulants Toutes les années")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admissibles':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admissibles, labels=['M: 65.61 %', 'F: 34.39 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admissibles Toutes les années")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Valides':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Valides, labels=['M: 64.42 %', 'F: 35.58 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Valides Toutes les années")
                                    plt.legend()

                                    st.pyplot(fig)
                                elif type_camaber == 'Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(Admis, labels=['M: 74.40 %', 'F: 25.60 %'], normalize=True)
                                    plt.title("Proportion M/F candidats Admis Toutes les années")
                                    plt.legend()

                                    st.pyplot(fig)

                        # -----------------------------------------------------------------------------------------------------------------------------------------
                        if choix == 2017:
                            if st.checkbox('Camamber - Unigenre'):
                                PostulantesVsAdmises = [2613, 111]
                                PostulantsVsAdmis = [4914, 473]
                                tendances = ['Filles Réfusées vs Filles Admises', 'Garçons Réfusés vs Garçons Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Filles Réfusées vs Filles Admises':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantesVsAdmises,
                                            labels=['Filles réfusées: 95.93 %', 'Filles Admises: 4.07 %'],
                                            normalize=True)
                                    plt.title("Proportion FR/FA candidats 2017")
                                    plt.legend()
                                    st.pyplot(fig)
                                elif type_camaber == 'Garçons Réfusés vs Garçons Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantsVsAdmis,
                                            labels=['Garçons réfusés: 91,22 %', 'Garçons Admis: 8,78 %'], normalize=True)
                                    plt.title("Proportion GR/GA candidats  2017")
                                    plt.legend()
                                    st.pyplot(fig)

                        elif choix == 2018:
                            if st.checkbox('Camamber - Unigenre'):
                                PostulantesVsAdmises = [2153, 150]
                                PostulantsVsAdmis = [3907, 448]
                                tendances = ['Filles Réfusées vs Filles Admises', 'Garçons Réfusés vs Garçons Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Filles Réfusées vs Filles Admises':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantesVsAdmises,
                                            labels=['Filles réfusées: 93.49 %', 'Filles Admises: 6.51 %'],
                                            normalize=True)
                                    plt.title("Proportion FR/FA candidats 2018")
                                    plt.legend()
                                    st.pyplot(fig)

                                elif type_camaber == 'Garçons Réfusés vs Garçons Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantesVsAdmises,
                                            labels=['Garçons réfusés: 89.71 %', 'Garçons Admis: 10.29 %'],
                                            normalize=True)
                                    plt.title("Proportion GR/GA candidats  2018")
                                    plt.legend()
                                    st.pyplot(fig)

                        elif choix == 2019:
                            if st.checkbox('Camamber - Unigenre'):
                                PostulantesVsAdmises = [2212, 191]
                                PostulantsVsAdmis = [3668, 531]
                                tendances = ['Filles Réfusées vs Filles Admises', 'Garçons Réfusés vs Garçons Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Filles Réfusées vs Filles Admises':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantesVsAdmises,
                                            labels=['Filles réfusées: 92.05 %', 'Filles Admises: 7.95 %'],
                                            normalize=True)
                                    plt.title("Proportion FR/FA candidats 2019")
                                    plt.legend()
                                    st.pyplot(fig)

                                elif type_camaber == 'Garçons Réfusés vs Garçons Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantsVsAdmis,
                                            labels=['Garçons réfusés: 87.25 %', 'Garçons Admis: 12.65 %'],
                                            normalize=True)
                                    plt.title("Proportion GR/GA candidats  2019")
                                    plt.legend()
                                    st.pyplot(fig)

                        elif choix == 2020:
                            if st.checkbox('Camamber - Unigenre'):
                                PostulantesVsAdmises = [2527, 203]
                                PostulantsVsAdmis = [4094, 452]
                                tendances = ['Filles Réfusées vs Filles Admises', 'Garçons Réfusés vs Garçons Admis']
                                type_camaber = st.selectbox('Choisir la tendance à srepresenter', tendances)
                                if type_camaber == 'Filles Réfusées vs Filles Admise':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantesVsAdmises,
                                            labels=['Filles réfusées: 92.56 %', 'Filles Admises: 7.44 %'],
                                            normalize=True)
                                    plt.title("Proportion FR/FA candidats 2020")
                                    plt.legend()
                                    st.pyplot(fig)

                                elif type_camaber == 'Garçons Réfusés vs Garçons Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantsVsAdmis,
                                            labels=['Garçons réfusés: 90.06 %', 'Garçons Admis: 9.94 %'], normalize=True)
                                    plt.title("Proportion GR/GA candidats  2020")
                                    plt.legend()
                                    st.pyplot(fig)

                        elif choix == 'Toutes les années':
                            if st.checkbox('Camamber - Unigenre'):
                                PostulantesVsAdmises = [9505, 655]
                                PostulantsVsAdmis = [16583, 1904]
                                tendances = ['Filles Réfusées vs Filles Admises', 'Garçons Réfusés vs Garçons Admis']
                                type_camaber = st.selectbox('Choisir la tendance à representer', tendances)
                                if type_camaber == 'Filles Réfusées vs Filles Admises':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantesVsAdmises,
                                            labels=['Filles réfusées: 93.55 %', 'Filles Admises: 6.45 %'],
                                            normalize=True)
                                    plt.title("Proportion FR/FA candidats ces quatres dernières années")
                                    plt.legend()
                                    st.pyplot(fig)

                                elif type_camaber == 'Garçons Réfusés vs Garçons Admis':
                                    fig = plt.figure(figsize=(8, 8))
                                    plt.pie(PostulantsVsAdmis,
                                            labels=['Garçons réfusés: 89.70 %', 'Garçons Admis: 10,30 %'],
                                            normalize=True)
                                    plt.title("Proportion GR/GA candidats ces quatres dernières années")
                                    plt.legend()
                                    st.pyplot(fig)


    elif choice == "ACP":

        st.subheader("ANALYSE DES COMPOSANTES PRINCIPALES")
        if data is not None:
            if data is not None:
                try:
                    df = pd.read_csv(data)
                    var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
                    # var_quali = df.select_dtypes(exclude=['float64', 'float32', 'double', 'int32', 'int64', 'date'])
                    all_columns = df.columns.to_list()
                except:
                    pass

                try:
                    df = pd.read_csv(data)
                    all_columns = df.columns.to_list()
                except:
                    pass

                try:
                    df = pd.read_table(data)
                    all_columns = df.columns.to_list()
                except:
                    pass

                if 'moyenneDossier' in all_columns and 'moyenneDefinitive' in all_columns:
                    if st.checkbox("Décorréler avec une ACP moyenneDossier et moyenneDefinitive"):
                        st.subheader("ACP sur deux variables")
                        try:
                            from sklearn.decomposition import PCA
                            d_values = df.loc[:, ['moyenneDossier', 'moyenneDefinitive']].values
                            moyDoss = d_values[:, 0]
                            moyDef = d_values[:, 1]
                            model_acp = PCA()
                            d_acp = model_acp.fit_transform(d_values)
                            comp1 = d_acp[:, 0]
                            comp2 = d_acp[:, 1]
                            coef_cor_comp, p_value_comp = pearsonr(comp1, comp2)
                            st.write(all_columns)

                            if st.checkbox("Nombre de composantes"):
                                st.write(model_acp.n_components_)
                            if st.checkbox("Les composantes"):
                                st.write(model_acp.components_)
                            if st.checkbox("Ratio de la variance expliquée"):
                                st.write(model_acp.explained_variance_ratio_)
                            if st.checkbox("Ratio de la variance expliquée cumulé croissant"):
                                st.write(model_acp.explained_variance_ratio_.cumsum())
                            if st.checkbox("Tester"):
                                st.write("Coéfficient de corrélation Composante 1 et 2 = ", coef_cor_comp,
                                         ' et p_value composante = ', p_value_comp)
                            if st.checkbox("Visualiser histogramme"):
                                fig = plt.figure()
                                plt.bar(range(model_acp.n_components_), model_acp.explained_variance_ratio_)
                                plt.xlabel("Composantes principales")
                                plt.ylabel("Pourcentage de variance expliquée")
                                plt.xticks(range(model_acp.n_components_))
                                plt.show()
                                st.pyplot(fig)
                            if st.checkbox("Visualiser droite de regression"):
                                fig, ax = plt.subplots()
                                ax.plot(comp1, comp2, linewidth=0, marker='s', label='Nuage de points')
                                ax.set_xlabel('comp1')
                                ax.set_ylabel('comp2')
                                ax.legend(facecolor='white')
                                plt.show()
                                st.pyplot(fig)

                        except:
                            pass
                    if st.checkbox("ACP sur 3 variables"):
                        # classe pour standardisation
                        from sklearn.preprocessing import StandardScaler
                        # instanciation
                        sc = StandardScaler()
                        # transformation–centrage-réduction
                        Z = sc.fit_transform(df.select_dtypes(include=['float', 'double']))
                        if st.checkbox('Centrer et reduire'):
                            st.write(Z)
                        if st.checkbox('Moyenne'):
                            st.table(np.mean(Z, axis=0))
                        if st.checkbox('ECart-type'):
                            st.table(np.std(Z, axis=0, ddof=0))
                        if st.checkbox('Afficher les paramètres'):
                            # classe pour l'ACP
                            from sklearn.decomposition import PCA
                            # instanciation
                            acp = PCA(svd_solver='full')
                            st.write(acp)
                        if st.checkbox('Nombre de composantes calculées'):
                            # calculs
                            coord = acp.fit_transform(Z)
                            # nombre de composantes calculées
                            st.write(acp.n_components_)

                        if st.checkbox('Variance expliquée'):
                            # variance expliquée
                            st.write(acp.explained_variance_)
                        if st.checkbox('Valeur corrigée'):
                            # valeur corrigée
                            n = acp.n_components_
                            eigval = (n - 1) / n * acp.explained_variance_
                            st.write(eigval)
                        if st.checkbox('Valeur singulière'):
                            # ou bien en passant par les valeurs singulières
                            n = acp.n_components_
                            st.write(acp.singular_values_ ** 2 / n)
                        if st.checkbox('Proportion de variance expliquée'):
                            # proportion de variance expliquée
                            st.write(acp.explained_variance_ratio_)
                        if st.checkbox('Eboulis des valeurs propres'):
                            # scree plot
                            fig = plt.figure()
                            n = acp.n_components_
                            eigval = (n - 1) / n * acp.explained_variance_
                            plt.plot(np.arange(1, n + 1), eigval)
                            plt.title("Scree plot")
                            plt.ylabel("Valeurs propres")
                            plt.xlabel("Nombre de valeurs propres")
                            plt.show()
                            st.pyplot(fig)
                        if st.checkbox('Cumul de valeurs propres expliquées'):
                            # cumul de variance expliquée
                            fig = plt.figure()
                            n = acp.n_components_
                            plt.plot(np.arange(1, n + 1), np.cumsum(acp.explained_variance_ratio_))
                            plt.title("Variances expliquée vs. Nombre de facteurs expliqués")
                            plt.ylabel("Cumul de variance-ratio expliquée")
                            plt.xlabel("Nombre de facteur")
                            plt.show()
                            st.pyplot(fig)
                        if st.checkbox('Seuil pour test des bâton brisés'):
                            # seuils pour test des bâtons brisés
                            bs = 1 / np.arange(n, 0, -1)
                            bs = np.cumsum(bs)
                            bs = bs[::-1]
                            st.write(pd.DataFrame({'Valeur Propre': eigval, 'Seuils': bs}))
                        if st.checkbox('Positionner les individus dans le premier plan'):
                            # positionnement des individus dans lepremier plan
                            fig, axes = plt.subplots(figsize=(12, 12))
                            axes.set_xlim(-6, 6)
                            # même limites en abscisse
                            axes.set_ylim(-6, 6)
                            # et en ordonnée
                            # #placement des étiquettes des observations
                            for i in range(n):
                                plt.annotate(df.select_dtypes(include=['float', 'double']).index[i],
                                             (coord[i, 0], coord[i, 1]))
                                # ajouter les axes
                                plt.plot([-6, 6], [0, 0], color='silver', linestyle='-', linewidth=1)
                                plt.plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
                                # affichage
                                plt.show()
                            st.pyplot(fig)

    elif choice == "LES TESTS STATISTIQUES":

        st.subheader("Quelques tests statistiques")
        if data is not None:
            if data is not None:
                try:
                    df = pd.read_csv(data)
                    var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
                    # var_quali = df.select_dtypes(exclude=['float64', 'float32', 'double', 'int32', 'int64', 'date'])
                    all_columns = df.columns.to_list()
                except:
                    pass

                try:
                    df = pd.read_csv(data)
                    all_columns = df.columns.to_list()
                except:
                    pass

                try:
                    df = pd.read_table(data)
                    all_columns = df.columns.to_list()
                except:
                    pass

                if 'moyenneDossier' in all_columns and 'moyenneDefinitive' in all_columns:
                    if st.checkbox("Vérifier la corrélation entre moyenne dossier et moyeene definitive"):
                        st.subheader(
                            "Sous l'hypothèse H0: <<les variables moyenneDossier et moyenneDefinitive sont corrélées contre H1:Il y a corrélation entre ces deux variables")
                        try:
                            d_values = df.loc[:, ['moyenneDossier', 'moyenneDefinitive']].values
                            moyDoss = d_values[:, 0]
                            moyDef = d_values[:, 1]
                            coef_cor, p_value = pearsonr(moyDoss, moyDef)
                            st.write(all_columns)
                            if st.checkbox("Tester"):
                                st.write("Corrélation(moyenneDossier,moyenneDefinitive) = ", coef_cor * 100.0,
                                         "et p_value = ", p_value)

                        except:
                            pass
                if 'moyenneDossier' in all_columns and 'moyenneDefinitive' in all_columns:
                    if st.checkbox("Regression Linéair moyenneDossier et moyenneDefinitive"):
                        try:
                            d_values = df.loc[:, ['moyenneDossier', 'moyenneDefinitive']].values
                            moyDoss = d_values[:, 0]
                            moyDef = d_values[:, 1]
                            st.write(all_columns)
                            slope, intercept, r, p, stderr = scipy.stats.linregress(moyDoss, moyDef)
                            line = f'moyenneDossier = {intercept:.2f}+{slope:.2f}.moyenneDefinitive, r={r:.2f}'
                            if st.checkbox("Droite de regression linéaire"):
                                st.write(line)
                            if st.checkbox("Visualiser droite de regression linéaire"):
                                fig, ax = plt.subplots()
                                ax.plot(moyDoss, moyDef, linewidth=0.5, marker='+', label='Points de données')
                                ax.plot(moyDoss, intercept + slope * moyDef, label=line)
                                ax.set_xlabel('Moyenne dossier')
                                ax.set_ylabel('Moyenne définitive')
                                ax.legend(facecolor='white')
                                plt.show()
                                st.write(fig)

                        except:
                            pass

                if st.checkbox("Contrôler la normalité des échantillons: Tets de Shapiro WIlk"):
                    all_columns = list(df.select_dtypes(include=['float', 'double']))
                    selected_columns = st.selectbox("Choisir la variable", all_columns)
                    st.subheader(
                        "Sous l'hypothèse H0: <<Distribution normale des données contre H1:Pas distribution normale")
                    stat, p = shapiro(df[selected_columns])
                    if st.checkbox("Tester"):
                        st.write("La Statistique =", stat, " p_value =", p)

                # if 'moyenneDossier' in all_columns and 'moyenneDefinitive' in all_columns:
                #     if st.checkbox("Contrôler l'égalité des variances : Fisher-Snedecor F-test"):
                #         st.subheader("Sous l'hypothèse H0: <<Il y a égalité de variance entre moyenneDossier et moyenneDefinitive H1:Pas d'égalité")
                #         try:
                #             d_values = df.loc[:,['moyenneDossier','moyenneDefinitive']].values
                #             moyDoss = d_values[:,0]
                #             moyDef = d_values[:,1]
                #             fTest = f(moyDoss, moyDef)
                #             st.write(all_columns)
                #             if st.checkbox("Tester"):
                #                 st.write("Statistique de Fisher =",fTest)
                #         except:
                #             pass

                # if 'moyenneDossier' in all_columns and 'moyenneDefinitive' in all_columns:
                #     if st.checkbox("ANOVA"):
                #         import scipy.stats as stats
                #         st.subheader("Sous l'hypothèse H0: <<Il y a égalité de variance entre moyenneDossier et moyenneDefinitive H1:Pas d'égalité")
                #         try:
                #             d_values = df.loc[:,['moyenneDossier','moyenneDefinitive']].values
                #             moyDoss = d_values[:,0]
                #             moyDef = d_values[:,1]
                #             anova = stats.f_oneway(moyDoss,moyDef)
                #             st.write(all_columns)
                #             if st.checkbox("Tester"):
                #                 st.write("Anova =",fTest)
                #         except:
                #             pass
                #

        # --------------------------------------------------FIN QUALITATIVES------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------QUELQUES GRAPHIQUES-----------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------------------------------------


#     elif choice == 'QUELQUES GRAPHIQUES':
#
#         st.subheader("Visualisation de nos données")
#         data = st.file_uploader("Charger une base de données", type=["csv", "txt", "xlsx"])
#         if data is not None:
#             try:
#                 df = pd.read_csv(data)
#                 st.dataframe(df.head())
#             except:
#                 st.warning("Un problème est survenu pendant le chargement de la base de données")
#
#
#
#             type_of_plot = st.selectbox("Choisir le type de graphique",
#                                         ["Bar chart", "Area Chart", "Line Chart", "Diagramme en ligne", "Diagramme en baton", "Histogramme", "Boxplot", "Densité", "Boite à Moustache"])
#             all_columns_names = df.columns.tolist()
#             selected_columns_names = st.multiselect("Choisir la table à representer", all_columns_names)
#
#             if st.button("Générez les graphiques"):
#                 try:
#                     st.success("Génération du graphe {} pour {} ...".format(type_of_plot, selected_columns_names))
#                 except:
#                     st.warning("Quelques chose à mal tourné !")
#
# #----------------------------------------------------------------------------------------------------------------------
#             if type_of_plot == "Bar Chart":
#                 try:
#                     cust_data = df[selected_columns_names]
#                     st.bar_chart(cust_data)
#                 except:
#                     pass
# # #----------------------------------------------------------------------------------------------------------------------
#                 if type_of_plot == 'Area Chart':
#                     try:
#                         cust_data = df[selected_columns_names]
#                         st.area_chart(cust_data)
#                     except:
#                         pass
# # #----------------------------------------------------------------------------------------------------------------------
# #
# #                 elif type_of_plot == 'Diagramme en bande':
# #                     try:
# #                         cust_data = df[selected_columns_names]
# #                         st.bar_chart(cust_data)
# #                     except:
# #                         st.warning("Impossible d'afficher ce graphique")
# # #----------------------------------------------------------------------------------------------------------------------
# #
# #                 elif type_of_plot == 'Diagramme en ligne':
# #                     try:
# #                         cust_data = df[selected_columns_names]
# #                         st.line_chart(cust_data)
# #                     except:
# #                         st.warning("Impossible d'afficher ce graphique")
# # #----------------------------------------------------------------------------------------------------------------------
# #
# #                 elif st.checkbox("Diagramme en bande de la population"):
# #                     try:
# #                         st.write(df.iloc[:, 'identifiantCandidat'].value_counts().plot(kind='bar'))
# #                         st.pyplot()
# #                     except:
# #                         st.warning("Impossible de répresenter ce graphique")
# # # ------------------------------------------------FIN GRAPHIQUES--------------------------------------------------------
# #
# # #-----------------------------------------------------------------------------------------------------------------------
# # #-------------------------------------------------Mise en forme des graphiques------------------------------------------
# # #-----------------------------------------------------------------------------------------------------------------------
#                 elif type_of_plot:
#                     try:
#                         cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
#                         st.write(cust_plot)
#                         st.pyplot()
#                     except:
#                         pass

if __name__ == '__main__':
    main()
