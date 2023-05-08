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
df.drop(['ID'], axis=1, inplace=True)
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
         df_fill[element] = scaler.fit_transform(df_fill[[element]])

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


###### Régression linéaire simple ######

# Créer un objet de modèle de régression linéaire
lin = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
lin.fit(x_train, y_train)

# Faire des prédictions sur les données de test
yLin_pred = lin.predict(x_test)



###### Régression Ridge ######

# Créer un objet de modèle de régression Ridge
Ridge = Ridge(alpha=0.1) 

# Entraîner le modèle sur les données d'entraînement
Ridge.fit(x_train, y_train)

# Faire des prédictions sur les données de test
yRidge_pred = Ridge.predict(x_test)



###### Régression Lasso ######

# Créer un objet de modèle de régression Lasso
Lasso = Lasso(alpha=0.1)

# Entraîner le modèle sur les données d'entraînement
Lasso.fit(x_train, y_train)

# Faire des prédictions sur les données de test
yLasso_pred = Lasso.predict(x_test)



###### Méthode régression k-NN ######

# Créer un objet de modèle de régression k-NN
knn = KNeighborsRegressor(n_neighbors=8)

# Entraîner le modèle sur les données d'entraînement
knn.fit(x_train, y_train)

# Faire des prédictions sur les données de test
yKNN_pred = knn.predict(x_test)



###### Arbre de décision pour la régression ######

# Créer un objet de modèle d'arbre de décision
reg = DecisionTreeRegressor()

# Entraîner le modèle sur les données d'entraînement
reg.fit(x_train, y_train)

# Faire des prédictions sur les données de test
yTree_pred = reg.predict(x_test)


#######################################
######  Evaluation des méthodes  ######
#######################################

from sklearn.metrics import mean_squared_error #evaluation metrics
from sklearn.metrics import r2_score #evaluation metrics
from scipy.stats import spearmanr


# Calculer l'erreur quadratique moyenne (RMSE)
print("\nRMSE pour la régression linéaire : ", np.sqrt(mean_squared_error(y_test, yLin_pred)))
print("RMSE pour la régression Ridge : ", np.sqrt(mean_squared_error(y_test, yRidge_pred)))
print("RMSE pour la régression Lasso : ", np.sqrt(mean_squared_error(y_test, yLasso_pred)))
print("RMSE pour la méthode des k-NN : ", np.sqrt(mean_squared_error(y_test, yKNN_pred)))
print("RMSE pour l'arbre de décision : ", np.sqrt(mean_squared_error(y_test, yTree_pred)))


# Calculer le coefficient de détermination (R2)
print("\nR2 pour la régression linéaire : ", r2_score(y_test, yLin_pred))
print("R2 pour la régression Ridge : ", r2_score(y_test, yRidge_pred))
print("R2 pour la régression Lasso : ", r2_score(y_test, yLasso_pred))
print("R2 pour la méthode des k-NN : ", r2_score(y_test, yKNN_pred))
print("R2 pour l'arbre de décision : ", r2_score(y_test, yTree_pred))


# Calculer la correlation de sperman
rho, pval = spearmanr(y_test, yLin_pred)
print("\nSpearman pour la régression linéaire : ")
print("rho : \n", rho)
print("pval : \n", pval)

rho, pval = spearmanr(y_test, yRidge_pred)
print("\nSpearman pour la régression Ridge : ")
print("rho : \n", rho)
print("pval : \n", pval)

rho, pval = spearmanr(y_test, yLasso_pred)
print("\nSpearman pour la régression Lasso : ")
print("rho : \n", rho)
print("pval : \n", pval)

rho, pval = spearmanr(y_test, yKNN_pred)
print("\nSpearman pour la méthode des k-NN : ")
print("rho : \n", rho)
print("pval : \n", pval)

rho, pval = spearmanr(y_test, yTree_pred)
print("\nSpearman pour l'arbre de décision : ")
print("rho : \n", rho)
print("pval \n: ", pval)

###################################################################
######  Optimisation des hyperparamètres pour la régression  ######
###################################################################

from sklearn.model_selection import GridSearchCV

# Créer un objet de modèle de régression Ridge
Ridge2 = Ridge()

# Créer un dictionnaire de valeurs d'hyperparamètres
param_grid = {'alpha': np.arange(0, 1, 0.1)}

# Créer un objet de recherche sur grille
Ridge_gscv = GridSearchCV(Ridge2, param_grid, cv=5)

# Ajuster l'objet de recherche sur grille aux données d'entraînement
Ridge_gscv.fit(x_train, y_train)

# Afficher les meilleurs paramètres après optimisation
print("Meilleurs paramètres pour la régression Ridge : ", Ridge_gscv.best_params_)
print("Meilleur score pour la régression Ridge : ", Ridge_gscv.best_score_)
print("Meilleur estimateur pour la régression Ridge : ", Ridge_gscv.best_estimator_)
print("Meilleur index pour la régression Ridge : ", Ridge_gscv.best_index_)
print("Meilleur score de validation croisée pour la régression Ridge : ", Ridge_gscv.best_score_)
print("Meilleur score de test pour la régression Ridge : ", Ridge_gscv.score(x_test, y_test))


# Créer un objet de modèle de régression Lasso
Lasso2 = Lasso()

# Créer un dictionnaire de valeurs d'hyperparamètres
param_grid = {'alpha': np.arange(0, 1, 0.1)}

# Créer un objet de recherche sur grille
Lasso_gscv = GridSearchCV(Lasso2, param_grid, cv=5)

# Ajuster l'objet de recherche sur grille aux données d'entraînement
Lasso_gscv.fit(x_train, y_train)

# Afficher les meilleurs paramètres après optimisation
print("\nMeilleurs paramètres pour la régression Lasso : ", Lasso_gscv.best_params_)
print("Meilleur score pour la régression Lasso : ", Lasso_gscv.best_score_)
print("Meilleur estimateur pour la régression Lasso : ", Lasso_gscv.best_estimator_)
print("Meilleur index pour la régression Lasso : ", Lasso_gscv.best_index_)
print("Meilleur score de validation croisée pour la régression Lasso : ", Lasso_gscv.best_score_)
print("Meilleur score de test pour la régression Lasso : ", Lasso_gscv.score(x_test, y_test))


# Créer un objet de modèle de régression k-NN
knn2 = KNeighborsRegressor()

# Créer un dictionnaire de valeurs d'hyperparamètres
param_grid = {'n_neighbors': np.arange(1, 25)}

# Créer un objet de recherche sur grille
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

# Ajuster l'objet de recherche sur grille aux données d'entraînement
knn_gscv.fit(x_train, y_train)

# Afficher les meilleurs paramètres après optimisation
print("\nMeilleurs paramètres pour la méthode des k-NN : ", knn_gscv.best_params_)
print("Meilleur score pour la méthode des k-NN : ", knn_gscv.best_score_)
print("Meilleur estimateur pour la méthode des k-NN : ", knn_gscv.best_estimator_)
print("Meilleur index pour la méthode des k-NN : ", knn_gscv.best_index_)
print("Meilleur score de validation croisée pour la méthode des k-NN : ", knn_gscv.best_score_)
print("Meilleur score de test pour la méthode des k-NN : ", knn_gscv.score(x_test, y_test))


# Créer un objet de modèle d'arbre de décision
tree2 = DecisionTreeRegressor()

# Créer un dictionnaire de valeurs d'hyperparamètres
param_grid = {'max_depth': np.arange(1, 25)}

# Créer un objet de recherche sur grille
tree_gscv = GridSearchCV(tree2, param_grid, cv=5)

# Ajuster l'objet de recherche sur grille aux données d'entraînement
tree_gscv.fit(x_train, y_train)

# Afficher les meilleurs paramètres après optimisation
print("\nMeilleurs paramètres pour l'arbre de décision : ", tree_gscv.best_params_)
print("Meilleur score pour l'arbre de décision : ", tree_gscv.best_score_)
print("Meilleur estimateur pour l'arbre de décision : ", tree_gscv.best_estimator_)
print("Meilleur index pour l'arbre de décision : ", tree_gscv.best_index_)
print("Meilleur score de validation croisée pour l'arbre de décision : ", tree_gscv.best_score_)
print("Meilleur score de test pour l'arbre de décision : ", tree_gscv.score(x_test, y_test))















