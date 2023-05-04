# Importation des librairies
import numpy as np #linear algebra
import pandas as pd #data manipulation and analysis
import seaborn as sns #data visualization
import matplotlib as matplotlib #data visualization
import matplotlib.pyplot as plt #data visualization
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler #data normalization
import sklearn.cluster as skc #machine learning (clustering)
import warnings # ignore warnings
warnings.filterwarnings('ignore')

# Importation du dataset
df = pd.read_csv('C:\Arthur\Efrei Paris\L3\Semestre 6\Intro apprentissage machine\Projet\projet_machine_learning\Data\Data_X.csv')


####################################
###### Péparation des données ######
####################################

# Affichage des statistiques du dataset
print("Statistique de df : ", df.describe()) 

# Affichage du nombre de valeur null par colonne
print("\nNbr valeur null : \n", df.isnull().sum()) 

#supprimer les lignes avec des valeurs 'DE' dans la colonne 'COUNTRY' ainsi que la colonne 'COUNTRY'
df1 = df[df.COUNTRY != 'DE']
df.drop(['COUNTRY'], axis=1, inplace=True)

#remplace les valeurs null par la moyenne de la colonne
df_fill = df.fillna(df.mean())

#Normalisation des données
scaler = MinMaxScaler() 
for element in df_fill.columns:
     if element != 'ID':
         df_fill[element] = scaler.fit_transform(df_fill[[element]])

print("\n\nDatas normalisées : \n", df_fill)



##################################################
###### Analyse exploratoire des donnees  #########
##################################################

###### APERCU DES VARIABLES ######

print("Distribution, plage de valeurs et signification des variables : \n")
print(df_fill.describe())

print("\n\nType des variable : \n")
print(df_fill.info())


###### RELATION ENTRE LES VARIABLES AVEC DES GRAPHIQUES ######

# Histogramme



# Diagramme en boite



# Graphiques de dispersion



####### MATRICE DE CORRELATION #######
# Calcul de la matrice de corrélation
correlation_metrics=df_fill.corr() 

# Generation d'un masque pour le triangle supérieur (en laissant la diagonale)
mask = np.zeros_like(correlation_metrics, dtype=bool)
mask[np.triu_indices_from(mask)] = True


# Setup de la figure matplotlib pour afficher la heatmap
fig = plt.figure(figsize=(18,9))

#crées un labels pour les valeurs = [-0.3;0.3]
Labels = (np.where(np.logical_and(correlation_metrics<0.3, correlation_metrics>-0.3),'',correlation_metrics.round(2)))

# Affichage des corrélations entre les variables avec une heatmap
sns.heatmap(correlation_metrics, cmap='RdBu', 
            vmax=1.0, vmin=-1.0, center=0, 
            fmt='', annot=Labels, 
            linewidths=.5,linecolor='black',
            cbar_kws={"shrink": .70}, mask=mask)

plt.title('Correlation Between Variables', size=14) 
# plt.show()





# plt.scatter(df['FR_TEMP'], df['DE_TEMP'], color = 'red')
# plt.title('FR_TEMP en fonction de DE_TEMP')
# plt.xlabel('température FR')
# plt.ylabel('température DE')
# plt.show()





