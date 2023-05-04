#Régression linaire simple

# Importation des librairies
import numpy as np #linear algebra
import pandas as pd #data manipulation and analysis
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #data visualization
import sklearn.preprocessing as skp #machine learning (preprocessing)
import sklearn.cluster as skc #machine learning (clustering)
import warnings # ignore warnings
warnings.filterwarnings('ignore')

# Importation du dataset
df = pd.read_csv('C:\Arthur\Efrei Paris\L3\Semestre 6\Intro apprentissage machine\Projet\projet_machine_learning\Data\DataNew_X.csv')

#affiche les statistiques du dataset
# print(df.describe()) 

#affiche le nombre de valeur null par colonne
# print("############# Nbr valeur null ############# \n", df.isnull().sum()) 


#remplace les valeurs null par la moyenne de la colonne
df_fill = df.fillna(df.mean())

# print(df_fill.describe())


# Calcul de la matrice de corrélation
correlation_metrics=df_fill.corr() 

# Generation d'un masque pour le triangle supérieur
mask = np.zeros_like(correlation_metrics, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Setup de la figure matplotlib
fig = plt.figure(figsize=(18,16))

#On enlève les valeurs qui sont en dessous de 0.3
labels = (np.where(correlation_metrics<0.3,'',correlation_metrics.round(2))) 

# Afiichaage des corrélations entre les variables avec une heatmap
sns.heatmap(correlation_metrics, cmap='RdBu', vmax=1.0, vmin=-1.0, center=0, fmt='', annot=labels, linewidths=1, cbar_kws={"shrink": .70}, mask=mask)
plt.title('Correlation Between Variables', size=14) 
plt.show()








# plt.scatter(df['FR_TEMP'], df['FR_SOLAR'], color = 'red')
# plt.title('FR_SOLAR en fonction de FR_TEMP')
# plt.xlabel('température FR')
# plt.ylabel('Photovoltaïque FR')
# plt.show()



