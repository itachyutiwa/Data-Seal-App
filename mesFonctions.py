# Core Pkgs
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

from unicodedata import normalize

from pandas import DataFrame


def asciize(df):
    return normalize("NFKD", df).encode("ascii", "ignore").decode("ascii")





#Chargement de données
def chargerDonnees():
    data = st.file_uploader("Chargez une base de données", type=["csv", "txt", "xlsx"])
    return data
#Lecture des données de type excel, csv ou txt

#-------------Information générales-------------------------

#Aperçu des données

def apercuDonnees(df):
        return st.write(df.head(10))


def dimensionDonnees(df):
    try:
        st.write(df.shape)
    except:
        st.warning("Un problème est survenu")

def Nan(df):
    try:
        return st.write(df.isna().sum())
    except:
        pass

# Pourcentage des Nan
def NaN_percent(df, column_name):
    column_name = df.columns.to_list()
    row_count = df[column_name].shape[0]
    empty_values = row_count - df[column_name].count()
    resultat = (100.0 * empty_values) / row_count
    return st.write(resultat)


def infoDonnees(df):
    try:
        st.write(df.info())
    except:
        st.warning("Rien à dire sure les identifiant")

def descriptionDonnees(df):

    try:
        df = df.select_dtypes(exclude=['int64', 'int32'])
        st.write(df.describe())
    except:
        st.warning("Rien à dire sure les identifiant")

def colonnesDonnees(df):
    try:
        all_columns = df.columns.to_list()
        st.write(all_columns)
    except:
        st.warning("Quelque chose ne va pas !")

def colonnesSelectionnees(df):
    try:
        all_columns = df.columns.to_list()
        selected_columns = st.multiselect("Choisissez la colonne", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)
    except:
        st.warning("Un problème est survenu")

#-----------------------Analyse univariée et multivariée
#-----------------Données quantitatives


def etenduQuanti(df):
    try:
        var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
        st.write(var_quanti.max() - var_quanti.min())
    except:
        st.warning("Rien à afficher")

def varianceQuanti(df):
    try:
        var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
        st.write((var_quanti.std()) ** 2)
    except:
        st.warning("Rien à afficher")

def coeffVariationQuanti(df):
    try:
        var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
        st.write(var_quanti.std() / var_quanti.mean())
    except:
        st.warning("Rien à afficher")

def intervQuanti(df):
        try:
            var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
            st.write(np.percentile(var_quanti, 25), ";", np.percentile(var_quanti, 75))
        except:
            st.warning("Rien à afficher")

def rapportVariationQuanti(df):
    try:
        var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
        st.write(var_quanti.max() / var_quanti.min())
    except:
        st.warning("Rien à afficher")

def nuagePointsQuanti(df):
    try:
        fig = plt.figure()
        var_quanti = df.select_dtypes(include=['float64', 'float32', 'double'])
        var_quanti.plot(x = var_quanti['moyenneDossier'], y= var_quanti['moyenneDefinitive'], kind = "scatter")
        st.pyplot(fig)
    except:
        st.warning("Rien à afficher")

#-------------------Données qualitatives

def effectifGenre(df):
    try:
        st.table(df.loc[:, "genre"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifNationalite(df):
    try:
        st.table(df.loc[:, "nationalite"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")
def effectifMention(df):
    try:
        st.table(df.loc[:, "mentionAuBac"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifRedoub(df):
    try:
        st.table(df.loc[:, "redoublementTerminale"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifFiliereAcc(df):
    try:

        st.table(df.loc[:, ["filiereAccueil","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifCycleAcc(df):
    try:
        st.table(df.loc[:, ["cycleAccueil","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def ecoleAcc(df):
    try:
        st.table(df.loc[:, ["ecoleAccueil","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifcycles(df):
    try:
        st.table(df.loc[:, ["cycles","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifcodeFilire(df):
    try:
        st.table(df.loc[:, "codeFiliere"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")



def effectifEtabOrigine(df):
    try:
        st.table(df.loc[:, "etabs"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifDrenGenre(df):
    try:
        st.table(df.loc[:, ["drens","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifFilereDren(df):
    try:
        st.table(df.loc[:, ["filiereAccueil","drens"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifEcoleAccDren(df):
    try:
        st.table(df.loc[:, ["ecoleAccueil","drens"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifMatiere(df):
    try:
        st.table(df.loc[:, "matiere"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifRangTerminale(df):
    try:
        st.table(df.loc[:, "rangTerminale"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifSerie(df):
    try:
        st.table(df.loc[:, "serie"].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifEtabOrigineGenre(df):
    try:
        st.table(df.loc[:, ["etabs","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifRangTerminaleGenre(df):
    try:
        st.table(df.loc[:, ["rangEnTerminale", "genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifSerieGenre(df):
    try:
        st.table(df.loc[:, ["serie","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifNationaliteGenre(df):
    try:
        st.table(df.loc[:, ["nationalite", "genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifDren(df):
    try:
        st.table(df.loc[:, 'drens'].value_counts())
    except:
        st.warning('Quelques chose ne va pas')

def effectifRangTerminaleDRen(df):
    try:
        st.table(df.loc[:, ["rangEnTerminale", "drens"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifSerieDRen(df):
    try:
        st.table(df.loc[:, ["serie", "drens"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def effectifDrenFilAccGenre(df):
    try:
        st.table(df.loc[:, ["drens", "filiereAccueil","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")
def effectifFilereGenre(df):
    try:
        st.table(df.loc[:, ["filiereAccueil","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def  effectifEcoleGenre(df):
    try:
        st.table(df.loc[:, ["ecoleAccueil","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")

def  effectifcodeFilireGenre(df):
    try:
        st.table(df.loc[:, ["codeFiliere","genre"]].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")


def effectifFiliereSerie(df):
    try:
        st.table(df.loc[:, ["filiereAccueil","serie"]].value_counts())
    except:
            st.warning("Quelques chose ne va pas !")
def Aberrantes(df):
    try:
        st.table(df[df["age"]>24].value_counts())
    except:
        st.warning("Quelques chose ne va pas !")


#||||||||||||||||||||----
# postulant = pd.read_csv('./STOCK/candidats_postulants.csv')
# admss = pd.read_csv('./STOCK/candidats_admissibles.csv')
# valid = pd.read_csv('./STOCK/candidats_valides.csv')
# admis = pd.read_csv('./STOCK/candidats_admis.csv')
# H = [effectifGenre(postulant[postulant['genre']=='M']),effectifGenre(admss[admss['genre'] == 'M']),
#      effectifGenre(valid[valid['genre'] == 'M']), effectifGenre(admis[admis['genre'] == 'M'])]
# F = [effectifGenre(postulant[postulant['genre']=='F']),effectifGenre(admss[admss['genre'] == 'F']),
#      effectifGenre(valid[valid['genre'] == 'F']), effectifGenre(admis[admis['genre'] == 'F'])]
# def nbHPostulant():
#     H = [effectifGenre(postulant[postulant['genre']=='M']),effectifGenre(admss[admss['genre'] == 'M']),
#      effectifGenre(valid[valid['genre'] == 'M']), effectifGenre(admis[admis['genre'] == 'M'])]
#     F = [effectifGenre(postulant[postulant['genre']=='F']),effectifGenre(admss[admss['genre'] == 'F']),
#      effectifGenre(valid[valid['genre'] == 'F']), effectifGenre(admis[admis['genre'] == 'F'])]

