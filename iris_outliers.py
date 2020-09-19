# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
import numpy as np 
import sklearn
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter

iris = pd.read_csv('iris.csv')
iris = iris.astype({"petallength": np.float, "petalwidth": np.float, "sepallength": np.float, "sepalwidth": np.float}) #standardizing types to float
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

#sb.pairplot(dataset, hue="class")
#plt.show()

# Analisis de datos - Deteccion de outliers para los atributos de las especies Iris sp 
# usando gráficos de caja y bigote 

def histograma_setosa(atrib):
    plt.title('Iris Setosa - ' + atrib)
    setosa = iris.loc[iris["class"] == "Iris-setosa"]
    plt.hist(setosa[atrib], edgecolor = "black", linewidth=1)
    plt.boxplot(setosa[atrib],vert=False)
    plt.show()
    Q1 = round(float(setosa[atrib].quantile(0.25)),2)
    Q3 = round(float(setosa[atrib].quantile(0.75)),2)
    IQ = round(float(Q3 - Q1),2)
    Med = float(setosa[atrib].median())
    Min = float(setosa[atrib].min())
    Max = float(setosa[atrib].max())
    LIF = round(float(Q1 - (1.5*IQ)),2)
    UIF = round(float(Q3 + (1.5*IQ)),2)
    LOF = round(float(Q1 - (3.0*IQ)),2)
    UOF = round(float(Q3 + (3.0*IQ)),2)
    print("LIF", LIF, "UIF", UIF, "LOF", LOF, "UOF", UOF)
    escalar_outliers = (setosa[atrib] < LIF) | (setosa[atrib] > UIF)
    outliers = setosa[escalar_outliers]
    print (outliers.sort_values(atrib))

def histograma_virginica(atrib):
    plt.title('Iris Virginica - ' + atrib)
    virginica = iris.loc[iris["class"] == "Iris-virginica"]
    plt.hist(virginica[atrib], edgecolor = "black", linewidth=1)
    plt.boxplot(virginica[atrib],vert=False)
    plt.show()
    Q1 = round(float(virginica[atrib].quantile(0.25)),2)
    Q3 = round(float(virginica[atrib].quantile(0.75)),2)
    IQ = round(float(Q3 - Q1),2)
    Med = float(virginica[atrib].median())
    Min = float(virginica[atrib].min())
    Max = float(virginica[atrib].max())
    LIF = round(float(Q1 - (1.5*IQ)),2)
    UIF = round(float(Q3 + (1.5*IQ)),2)
    LOF = round(float(Q1 - (3.0*IQ)),2)
    UOF = round(float(Q3 + (3.0*IQ)),2)

    print("LIF", LIF, "UIF", UIF, "LOF", LOF, "UOF", UOF)
    escalar_outliers = (virginica[atrib] < LIF) | (virginica[atrib] > UIF)
    outliers = virginica[escalar_outliers]
    print (outliers.sort_values(atrib))

def histograma_versicolor(atrib):
    plt.title('Iris Versicolor -' + atrib)
    versicolor = iris.loc[iris["class"] == "Iris-versicolor"]
    plt.hist(versicolor[atrib], edgecolor = "black", linewidth=1)
    plt.boxplot(versicolor[atrib],vert=False)
    plt.show()
    Q1 = round(float(versicolor[atrib].quantile(0.25)),2)
    Q3 = round(float(versicolor[atrib].quantile(0.75)),2)
    IQ = round(float(Q3 - Q1),2)
    Med = float(versicolor[atrib].median())
    Min = float(versicolor[atrib].min())
    Max = float(versicolor[atrib].max())
    LIF = round(float(Q1 - (1.5*IQ)),2)
    UIF = round(float(Q3 + (1.5*IQ)),2)
    LOF = round(float(Q1 - (3.0*IQ)),2)
    UOF = round(float(Q3 + (3.0*IQ)),2)

    print("LIF", LIF, "UIF", UIF, "LOF", LOF, "UOF", UOF)
    escalar_outliers = (versicolor[atrib] < LIF) & (versicolor[atrib] > UIF)
    outliers = versicolor[escalar_outliers]
    print (outliers.sort_values(atrib))

histograma_setosa("sepallength")
histograma_setosa("sepalwidth")
histograma_setosa("petallength")
histograma_setosa("petalwidth")

histograma_virginica("sepallength")
histograma_virginica("sepalwidth")
histograma_virginica("petallength")
histograma_virginica("petalwidth")

histograma_versicolor("sepallength")
histograma_versicolor("sepalwidth")
histograma_versicolor("petallength")
histograma_versicolor("petalwidth")


iris.columns=['Alto - Sepalo', 'Ancho - Sepalo', 'Alto - Pétalo', 'Ancho - Pétalo', 'Especie']
data = iris.iloc[:,0:4]
target = iris.iloc[:,4]

#labels = np.array(iris['class'])
model= DBSCAN(eps=0.7, min_samples=19, leaf_size=30, metric='euclidean', algorithm='auto', n_jobs=1, p=None).fit(data)

print('------- Outliers encontrados usando DBSCAN  -----------')
print(model)
outliers_df = pd.DataFrame(data)
print (Counter(model.labels_))
print (outliers_df[model.labels_==-1])

fig = plt.figure()
ax = fig.add_axes([.1,.1, 1,1])
colors = model.labels_
ax.scatter(data.iloc[:,2].values, data.iloc[:,1].values, c=colors, s=120)
ax.set_xlabel('Length')
ax.set_ylabel('Width')
plt.title('Comparativa de detección de Outliers usando DBSCAN')


# %%



