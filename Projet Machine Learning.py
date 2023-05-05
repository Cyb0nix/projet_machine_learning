# Importation des librairies
import numpy as np #linear algebra
import pandas as pd #data manipulation and analysis
import seaborn as sns #data visualization
import matplotlib as matplotlib #data visualization
import matplotlib.pyplot as plt #data visualization
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler #data normalization
from sklearn.model_selection import train_test_split #data split
from sklearn.metrics import mean_squared_error #evaluation metric
import sklearn.cluster as skc #machine learning (clustering)
import warnings # ignore warnings
warnings.filterwarnings('ignore')

# Importation du dataset
df = pd.read_csv('C:\Arthur\Efrei Paris\L3\Semestre 6\Intro apprentissage machine\Projet\projet_machine_learning\Data\Data_X.csv')
Y = pd.read_csv('C:\Arthur\Efrei Paris\L3\Semestre 6\Intro apprentissage machine\Projet\projet_machine_learning\Data\Data_Y.csv')

####################################
###### Péparation des données ######
####################################

# Affichage des statistiques du dataset
print("Statistique de df : ", df.describe()) 

# Affichage du nombre de valeur null par colonne
print("\nNbr valeur null : \n", df.isnull().sum()) 

#supprimer varable inutile
df.drop(['COUNTRY'], axis=1, inplace=True) 
df.drop(['ID'], axis=1, inplace=True) 
df.drop(['DAY_ID'], axis=1, inplace=True) 
df.drop(['FR_TEMP'], axis=1, inplace=True) 
df.drop(['DE_TEMP'], axis=1, inplace=True) 
df.drop(['FR_RAIN'], axis=1, inplace=True)
df.drop(['DE_RAIN'], axis=1, inplace=True)
df.drop(['GAS_RET'], axis=1, inplace=True)
df.drop(['COAL_RET'], axis=1, inplace=True)
df.drop(['CARBON_RET'], axis=1, inplace=True)
df.drop(['DE_FR_EXCHANGE'], axis=1, inplace=True)
df.drop(['FR_DE_EXCHANGE'], axis=1, inplace=True)

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
# df_fill.hist(figsize=(18,9))


# # Diragramme en boite
# fig, axes = plt.subplots(7, 5, figsize=(18, 9)) # 7 rows, 5 columns

# for i, col in enumerate(df_fill.describe().columns): # enumerate() returns a tuple containing a count (from start which defaults to 0) and the values obtained from iterating over df.columns
#     ax = axes[i//5, i%5]
#     sns.boxplot(x=df_fill[col], ax=ax)
#     ax.set_title('{}'.format(col))
#     plt.tight_layout()


# # Graphiques de dispersion
# pd.plotting.scatter_matrix(df_fill, figsize=(18,9))


####### MATRICE DE CORRELATION #######
# Calcul de la matrice de corrélation
# correlation_metrics=df_fill.corr() 

# # Generation d'un masque pour le triangle supérieur (en laissant la diagonale)
# mask = np.zeros_like(correlation_metrics, dtype=bool)
# mask[np.triu_indices_from(mask)] = True

# # Setup de la figure matplotlib pour afficher la heatmap
# fig = plt.figure(figsize=(18,9))

# #crées un labels pour les valeurs = [-0.3;0.3]
# Labels = (np.where(np.logical_and(correlation_metrics<0.3, correlation_metrics>-0.3),'',correlation_metrics.round(2)))

# # Affichage des corrélations entre les variables avec une heatmap
# sns.heatmap(correlation_metrics, cmap='RdBu', 
#             vmax=1.0, vmin=-1.0, center=0, 
#             fmt='', annot=Labels, 
#             linewidths=.5,linecolor='black',
#             cbar_kws={"shrink": .70}, mask=mask)

# plt.title('Correlation Between Variables', size=14) 
# plt.show()


###### INTERPRETATION DES RESULTATS DE L'EDA ######
# TODO: Interpréter les résultats de l’EDA pour identifier les caractéristiques importantes qui influencent le prix de l’électricité et les relations significatives entre les variables


#######################################
###### MODELISATION DES DONNEES  ######
#######################################
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


x_train, x_test, y_train, y_test = train_test_split(df_fill, Y, test_size=0.25, random_state=21, stratify=Y)

print("Dimension x_train : "+str(x_train.shape))
print("Dimension x_test : "+str(x_test.shape))

print("Dimension y_train : "+str(y_train.shape))
print("Dimension y_test : "+str(y_test.shape))


# Régression linéaire simple



# Régression Ridge



# Régression Lasso




# Méthode des k-NN



# Arbre de décision pour la régression










