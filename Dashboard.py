# https://glsm-p7-openclassroom-dash-streamlit-sg9fg7.streamlit.app

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import pickle
from PIL import Image
import plotly.graph_objects as go 
import math
from urllib.request import urlopen
import json
import requests
import joblib


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

import warnings
warnings.filterwarnings("ignore")

with open('style.css') as f:
   st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


#Chargement des données
PATH = './'
df = pd.read_csv(PATH+'data_test.csv')
data_train =pd.read_parquet(PATH+'application_train.parquet')
 #pd.read_pickle(PATH+'application_train.pkl')
data_test = pd.read_parquet(PATH+'application_test.parquet')
#pd.read_pickle(PATH+'app_test.pkl')
description = pd.read_csv(PATH+'HomeCredit_columns_description.csv', 
                                      usecols=['Row', 'Description'], \
                                  index_col=0, encoding='unicode_escape')

#Chargement du modèle
model = joblib.load(PATH+r"LGBMClassifier.joblib")


#Les fonctions communes

@st.cache
def get_client_info(data, id_client):
    client_info = data[data['SK_ID_CURR']==int(id_client)]
    return client_info

@st.cache(suppress_st_warning=True)

def plot_distribution(applicationDF,feature, client_feature_val, title):

    if (not (math.isnan(client_feature_val))):
        fig = plt.figure(figsize = (10, 4))

        t0 = applicationDF.loc[applicationDF['TARGET'] == 0]
        t1 = applicationDF.loc[applicationDF['TARGET'] == 1]

        if (feature == "DAYS_BIRTH"):
            sns.kdeplot((t0[feature]/-365).dropna(), label = 'Payé', color='g')
            sns.kdeplot((t1[feature]/-365).dropna(), label = 'Défaillant', color='r')
            plt.axvline(float(client_feature_val/-365),  color="blue", linestyle='--', label = 'Position Client')

        elif (feature == "DAYS_EMPLOYED"):
            sns.kdeplot((t0[feature]/365).dropna(), label = 'Payé', color='g')
            sns.kdeplot((t1[feature]/365).dropna(), label = 'Défaillant', color='r')    
            plt.axvline(float(client_feature_val/365), color="blue", linestyle='--', label = 'Position Client')

        else:    
            sns.kdeplot(t0[feature].dropna(), label = 'Payé', color='g')
            sns.kdeplot(t1[feature].dropna(), label = 'Défaillant', color='r')
            plt.axvline(float(client_feature_val), color="blue",linestyle='--', label = 'Position Client')


        plt.title(title, fontsize='20', fontweight='bold')
        plt.legend()
        plt.show()  
        st.pyplot(fig)
    else:
        st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")

@st.cache(suppress_st_warning=True)

def univariate_categorical(applicationDF,feature,client_feature_val,titre,ylog=False,label_rotation=False, horizontal_layout=True):
        if (client_feature_val.iloc[0] != np.nan):

            temp = applicationDF[feature].value_counts()
            df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

            categories = applicationDF[feature].unique()
            categories = list(categories)

            # Calculate the percentage of target=1 per category value
            
            cat_perc = applicationDF[[feature,'TARGET']].groupby([feature],as_index=False).mean()
            cat_perc["TARGET"] = cat_perc["TARGET"]*100
            cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

            if(horizontal_layout):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))

            # 1. Subplot 1: Count plot of categorical column
            sns.set_palette("Set2")
            s = sns.countplot(ax=ax1, 
                            x = feature, 
                            data=applicationDF,
                            hue ="TARGET",
                            order=cat_perc[feature],
                            palette=['g','r'])

            pos1 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
            #st.write(client_feature_val.iloc[0])

            # Define common styling
            ax1.set(ylabel = "Nombre de clients")
            ax1.set_title(titre, fontdict={'fontsize' : 15, 'fontweight' : 'bold'})   
            ax1.axvline(int(pos1), color="blue", linestyle='--', label = 'Position Client')
            ax1.legend(['Position Client','Payé','Défaillant' ])

            # If the plot is not readable, use the log scale.
            if ylog:
                ax1.set_yscale('log')
                ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 15,'fontweight' : 'bold'})   
            if(label_rotation):
                s.set_xticklabels(s.get_xticklabels(),rotation=90)

            # 2. Subplot 2: Percentage of defaulters within the categorical column
            s = sns.barplot(ax=ax2, 
                            x = feature, 
                            y='TARGET', 
                            order=cat_perc[feature], 
                            data=cat_perc,
                            palette='Set2')

            pos2 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])

            if(label_rotation):
                s.set_xticklabels(s.get_xticklabels(),rotation=90)
            plt.ylabel('Pourcentage de défaillants [%]', fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            ax2.set_title(titre+" (% Défaillants)",fontdict={'fontsize' : 15, 'fontweight' : 'bold'})
            ax2.axvline(int(pos2), color="blue", linestyle='--', label = 'Position Client')
            ax2.legend()
            plt.show()
            st.pyplot(fig)
        else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")
            
ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
relevant_features = [col for col in df if col not in ignore_features]


## SideBar

with st.sidebar:
    image = Image.open('pret.png')
    st.image(image, use_column_width=True)
   
    st.write("## N° client")
    id_list = df["SK_ID_CURR"].values
    id_client = st.selectbox("Sélectionner l'ID du client", id_list)

    st.write("## Veuillez selectionner a vos actions")

    show_credit_decision = st.checkbox("La décision de crédit")
    show_client_details = st.checkbox("Les informations du client")
    show_client_comparison = st.checkbox("Comparer aux autres clients")
    fi_general = st.checkbox("La feature importance globale")

 

### # Main page 🎈

main_page = """
    <div style="background-color: #36b9cc; padding:10px; border-radius:5px">
    <h1 style="color: white; text-align:center">Scoring Crédit</h1> </div>
    
      ***
    <p style="font-size: 20px; font-weight: bold; text-align:center">
    Tableau de bord interactif pour les responsables de la relation client </p>
    """
st.markdown(main_page, unsafe_allow_html=True)


st.write('--*Ce tableau de bord est produit à partir du modèle créé à partir des données historiques des clients.*-')

#N° client Client sélectionné
st.write("N° client Sélectionné :", id_client)



if (int(id_client) in id_list):
    client_info = data_test[data_test['SK_ID_CURR']==int(id_client)]
    

#La décision 
        
    if (show_credit_decision):
        st.header('‍ Le score et la décision du modèle de crédit')
            
#Appel de l'API : 

        API_url = "https://p7opencgk.herokuapp.com/credit/" + str(id_client)

        with st.spinner('Chargement du score du client...'):
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())
            classe_predite = API_data['prediction']
            if classe_predite == 1:
                decision = "<font color='red'> **Refus de la demande de prêt pour le client** </font>"
            else:
                decision = "<font color='green'> **Acceptation de la demande de prêt pour le client** </font>"

            proba = 1-API_data['proba']
            client_score = round(proba*100, 2)

            col1, col2 = st.columns([3,3])

            gauge = go.Figure(go.Indicator(
             domain = {'x': [0, 1], 'y': [0, 1]},
             value = client_score,
             mode = "gauge+number+delta",
             title = {'text': "Pourcentage de risque de défaut"},
             gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 50], 'color': "white"},
                 {'range': [50, 100], 'color': "red"}],
             'threshold' : {'line': {'color': "black", 'width': 10}, 'thickness': 0.75, 'value': client_score}}))

            gauge.update_layout(width=450, height=200, margin=dict(l=50, r=50, b=0, t=0, pad=4))
            col1.plotly_chart(gauge)

            col2.markdown('Risque de défaut: **{}%**'.format(str(client_score)))
          
            if classe_predite == 1:
                col2.markdown('Décision: <span style="color:red">**{}**</span>'.format(decision), unsafe_allow_html=True)   
            else:    
                col2.markdown('Décision: <span style="color:green">**{}**</span>'.format(decision),unsafe_allow_html=True)


#Local Feature Importance

        show_local_feature_importance = st.checkbox(
                "Afficher les variables ayant le plus contribué à la décision du modèle ?")
        if (show_local_feature_importance):
            shap.initjs()
            number = st.slider('Sélectionner le nombre de feautures à afficher ?',2, 20, 8)

            X = df[df['SK_ID_CURR']==int(id_client)]
            X = X[relevant_features]

            fig, ax = plt.subplots(figsize=(15, 15))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.summary_plot(shap_values[0], X, plot_type ="bar",max_display=number, color_bar=True, plot_size=(4, 4))

            st.pyplot(fig)
            
    personal_info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "POSSESSION VEHICULE",
            'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "EMPLOI",
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "EXT_SOURCE_1",
            'EXT_SOURCE_2': "EXT_SOURCE_2",
            'EXT_SOURCE_3': "EXT_SOURCE_3",

        }

    default_list= ["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","REVENUS","MONTANT CREDIT"]
    numerical_features = [ 'DAYS_BIRTH' ,'CNT_CHILDREN','DAYS_EMPLOYED','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1', 
                         'EXT_SOURCE_2 ' , 'EXT_SOURCE_3' ]

    rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
    horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

    if (show_client_details):
        st.header('‍ Informations relatives au client')

        with st.spinner('Chargement des informations relatives au client...'):

            personal_info_df = client_info[list(personal_info_cols.keys())]
               
            personal_info_df.rename(columns=personal_info_cols, inplace=True)

            personal_info_df["AGE"] = int(round(personal_info_df["AGE"]/365*(-1)))
            personal_info_df["NB ANNEES EMPLOI"] = int(round(personal_info_df["NB ANNEES EMPLOI"]/365*(-1)))


            filtered = st.multiselect("Choisir les informations à afficher",options=list(personal_info_df.columns),
                                      default=list(default_list))
            df_info = personal_info_df[filtered] 
            df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
            df_info = df_info.set_index('SK_ID_CURR')

            st.table(df_info.astype(str).T)
            show_all_info = st.checkbox("Afficher toutes les informations")
            if (show_all_info):
                st.dataframe(client_info)



#------------------------------------------------------

    if (show_client_comparison):
        st.header('‍ Comparaison aux autres clients')
            #st.subheader("Comparaison avec l'ensemble des clients")
      
            
        with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):

            var = st.selectbox("Sélectionner une variable",list(personal_info_cols.values()))

            feature = list(personal_info_cols.keys())[list(personal_info_cols.values()).index(var)]
            
            if (feature in numerical_features):
                
                 plot_distribution(data_train, feature, client_info[feature], var)  
                
            elif (feature in rotate_label):
                
                 univariate_categorical(data_train, feature,client_info[feature], var, False, True)
            elif (feature in horizontal_layout):
               
                
                 univariate_categorical(data_train, feature,client_info[feature], var, False, True, True)
                
            else:
                
                 univariate_categorical(data_train, feature, client_info[feature], var)

       
    if (fi_general):
        st.header('‍Feature importance globale')
        st.image('global_feature_imp.png')

fi_general = "global_feature_imp.png"

st.sidebar.markdown('''
---
Created by GLSM
''')