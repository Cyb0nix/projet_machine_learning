# Importation des librairies
import numpy as np #linear algebra
import pandas as pd #data manipulation and analysis
import seaborn as sns #data visualization
import matplotlib as matplotlib #data visualization
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.discriminant_analysis import StandardScaler
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler #data normalization
from sklearn.model_selection import train_test_split #data split
import sklearn.cluster as skc #machine learning (clustering)
import warnings # ignore warnings
warnings.filterwarnings('ignore')

# Importation du dataset
df = pd.read_csv('Data\Data_X.csv')
Y = pd.read_csv('Data\Data_Y.csv')

####################################
###### Péparation des données ######
####################################

# Affichage des statistiques du dataset
print("Statistique de df : ", df.describe()) 

# Affichage du nombre de valeur null par colonne
print("\nNbr valeur null : \n", df.isnull().sum()) 

#supprimer varable inutile
df.drop(['COUNTRY'], axis=1, inplace=True) 
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
# scaler = MinMaxScaler() 

# print("\n\nDatas normalisées : \n", df_fill)



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
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

x_train, x_test, y_train, y_test = train_test_split(df_fill, Y, test_size=0.3, random_state=21)


###### Normalisation des données ######

# Initialize the StandardScaler function
df_fill = StandardScaler()

# Fit the StandardScaler on the trainig set
df_fill.fit(x_train)

# Standardization of the training set
X_train_norm = df_fill.transform(x_train)

# Standardization of the validation set
X_test_norm = df_fill.transform(x_test)



###### Régression linéaire simple ######

# Créer un objet de modèle de régression linéaire
lin = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
lin.fit(X_train_norm, y_train)

# Faire des prédictions sur les données de test
y_lin_pred = lin.predict(X_test_norm)
print("Linear score : ", lin.score(X_test_norm, y_test))



###### Régression Ridge ######

# Créer un objet de modèle de régression Ridge
Ridge = Ridge(alpha=1) 

# Entraîner le modèle sur les données d'entraînement
Ridge.fit(X_train_norm, y_train)

# Faire des prédictions sur les données de test
yRidge_pred = Ridge.predict(x_test)
print("Ridge score : ", Ridge.score(x_test, y_test))


###### Régression Lasso ######

# Créer un objet de modèle de régression Lasso
Lasso = Lasso(alpha=1)

# Entraîner le modèle sur les données d'entraînement
Lasso.fit(X_train_norm, y_train)

# Faire des prédictions sur les données de test
yLasso_pred = Lasso.predict(X_test_norm)
print("Lasso score : ", Lasso.score(X_test_norm, y_test))



###### Méthode régression k-NN ######

# Créer un objet de modèle de régression k-NN
knn = KNeighborsRegressor(n_neighbors=5)

# Entraîner le modèle sur les données d'entraînement
knn.fit(x_train, y_train)

# Faire des prédictions sur les données de test
yKNN_pred = knn.predict(x_test)
print("k-NN score : ", knn.score(x_test, y_test))






###### Arbre de décision pour la régression ######

# Créer un objet de modèle d'arbre de décision
reg = DecisionTreeRegressor()

# Entraîner le modèle sur les données d'entraînement
reg.fit(x_train, y_train)

# Faire des prédictions sur les données de test
yTree_pred = reg.predict(x_test)
print("Decision Tree score : ", reg.score(x_test, y_test))


#######################################
###### Evaluation des méthodes  ######
#######################################

from sklearn.metrics import mean_squared_error #evaluation metrics
from sklearn.metrics import r2_score #evaluation metrics
from scipy.stats import spearmanr


# Calculer l'erreur quadratique moyenne (RMSE)
# print("RMSE pour la régression linéaire : ", np.sqrt(mean_squared_error(y_test, y_lin_pred)))
print("RMSE pour la régression Ridge : ", np.sqrt(mean_squared_error(y_test, yRidge_pred)))
print("RMSE pour la régression Lasso : ", np.sqrt(mean_squared_error(y_test, yLasso_pred)))
# print("RMSE pour la méthode des k-NN : ", np.sqrt(mean_squared_error(y_test, yKNN_pred)))
# print("RMSE pour l'arbre de décision : ", np.sqrt(mean_squared_error(y_test, yTree_pred)))


# Calculer le coefficient de détermination (R2)
# print("R2 pour la régression linéaire : ", r2_score(y_test, y_lin_pred))
print("R2 pour la régression Ridge : ", r2_score(y_test, yRidge_pred))
print("R2 pour la régression Lasso : ", r2_score(y_test, yLasso_pred))
# print("R2 pour la méthode des k-NN : ", r2_score(y_test, yKNN_pred))
# print("R2 pour l'arbre de décision : ", r2_score(y_test, yTree_pred))


# Calculer la correlation de sperman
from scipy.stats import spearmanr
print("Spearman pour la régression linéaire : ", spearmanr(y_test, y_lin_pred))
print("Spearman pour la régression Ridge : ", spearmanr(y_test, yRidge_pred))
print("Spearman pour la régression Lasso : ", spearmanr(y_test, yLasso_pred))
# print("Spearman pour la méthode des k-NN : ", spearmanr(y_test, yKNN_pred))
# print("Spearman pour l'arbre de décision : ", spearmanr(y_test, yTree_pred))












