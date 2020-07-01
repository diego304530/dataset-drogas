#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd       
import numpy as np 
import math
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from math import e
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[18]:


# Se procedio llamar los datos guardados en la plataforma GIT-HUB
datos = pd.read_csv("https://raw.githubusercontent.com/diego304530/dataset-drogas/master/encuestaBinaria.csv", error_bad_lines=False)
datos.drop([ 'Unnamed: 0'], axis=1, inplace=True)


# In[19]:


datos.head(5)


# In[20]:


# Procedemos a cambiar el unico dato de tipo float al tipo de dato entero
datos.A_que_edad = datos.A_que_edad.astype(int)
datos.dtypes


# In[21]:


#EJECUTAMOS EL COMANDO PARA OBSERVAR CUANTAS PERSONAS CONSUMEN DROGAS
print(datos.groupby('Consume_droga').size())


# In[22]:


# graficamos las variables para seleccionar las que tengan menos interaccion 
#de respuesta y disminuir las variables para una mayor eficacia de los modelos 
plt.rcParams['figure.figsize']= [15.,25]
datos.drop([],1).hist()
plt.show()


# In[23]:


# eliminamos las variables que no aportan informacion al modelo y las eliminamos
datos.drop(['A_que_edad', 'Agresivo','Calle', 'Casa', 'Colegio.1','Con_pareja_sentimental', 
            'Continua', 'Impulsivo','Parque','Paseo', 'Sexualidad', 'Viudo_viuda', 'Solo Padre','No_consumido','Fiestas'], axis=1 ,inplace=True)


# In[24]:


# se realizara la correlacion de los datos para identificar los datos que no sirven para realizar el modelo 
#de prediccion y conocer
# cuales son los datos con mayor correlacion con la variable dependiente Consume_droga
def calcular_pvalue(datos):
    datos = datos.dropna()._get_numeric_data()
    datoscols = pd.DataFrame(columns=datos.columns)
    pvalues = datoscols.transpose().join(datoscols, how='outer')
    for r in datos.columns:
        for c in datos.columns:
            pvalues[r][c] = round(pearsonr(datos[r], datos[c])[1], 4)
    return pvalues
pval = calcular_pvalue(datos) 
# Vemos la correlacion y su significancia
rho = datos.corr()

rho = rho.round(3)

# create three masks
r1 = rho.applymap(lambda x: '{}*'.format(x))
r2 = rho.applymap(lambda x: '{}**'.format(x))
r3 = rho.applymap(lambda x: '{}***'.format(x))

# apply them where appropriate
rho = rho.mask(pval<=0.1,r1)
rho = rho.mask(pval<=0.05,r2)
rho = rho.mask(pval<=0.01,r3)
rho


# In[25]:


# se procedio a  eliminar las varibales independientes que tienen una correlacion alta entre ellas para que no afecten los modelos de prediccion 
datos.drop(['Grado','Relacion_familiar','Conocimiento_amistades_padres','Estrato'], axis=1, inplace=True)


# In[26]:


# realizamos una grafica de colores que indican segun su color la correlacion 
# de las variables independientes con la variable dependiente

plt.rcParams['figure.figsize'] = (20.0, 10.0)
corr = datos.corr()

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right');


# In[35]:


datos.drop(['Recreo'], axis=1, inplace=True)


# In[36]:



#FEATURE SELECTION

# columnas independientes
X = datos.iloc[:, datos.columns !='Consume_droga']

# Columna objetivo
y = datos.Consume_droga 

model = ExtraTreesClassifier()
model.fit(X,y)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show(1)


# In[16]:


# LUEGO DE SELECIONAR LAS VARIABLES MAS FACTIBLES PARA LA PREDICCION LAS UTILZAMOS PARA INCORPORARLAS
#EN LOS DIFERENTES MODELOS 
feature_names = ['Cigarrillo','Invitacion_drogas','Familiares_consumidores','Edad','Rupturas_amorosas']
X = datos[feature_names]
Y = datos['Consume_droga']
dataframe = datos
array = dataframe.values
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
# models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=6, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('COMPARACION DE ALGORITMOS')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[83]:


model = linear_model.LogisticRegression()
model.fit(X,Y)
print("precision de predicciones = ",model.score(X,Y))
validation_size = 0.70
from sklearn.model_selection import train_test_split

x_train,x_validation,y_train,y_validation = model_selection.train_test_split(X,Y, test_size = validation_size )
nombre= "regresion logistica"
muestra = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model,x_train, y_train, cv=muestra, scoring= 'accuracy')
# msg = "%s: %f (%f)" %(nombre,cv_SSSSresults.mean(), cv_results.std())
#print(msg)

predictions = model.predict(x_validation)
print("exactitud de predicion : ", accuracy_score(y_validation,predictions))
#print(confusion_matrix(y_validation,predictions))


# In[41]:


def algoritmo(a,b,c,d,e):
    import pandas
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import pickle
    import pandas as pd
    from sklearn.metrics import classification_report
    url = "https://raw.githubusercontent.com/diego304530/dataset-drogas/master/encuestaBinaria.csv"
    names = ['Cigarrillo','Invitacion_drogas','Familiares_consumidores','Edad','Rupturas_amorosas']
    dataframe = pandas.read_csv(url)
    array = dataframe.values
    X = dataframe[names] # variables independientes seleccionadas con feature selection
    Y = array[:,13] # Posicion de la variable dependiente
    test_size = 0.9
    seed = 7
    validation_size = 0.70
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size = validation_size )
    # Fit the model on training set
    model = LogisticRegression()
    model.fit(X, Y)

    result = model.score(X_test, Y_test)

    DatosConsola = pd.DataFrame({'Cigarrillo':[a],'Invitacion_drogas':[b],'Familiares_consumidores':[c], 'Edad':[d],
                                 'Rupturas_amorosas':[e],})
    print("posible a consumir droga", int(model.predict(DatosConsola)), " porcentaje de acierto: ", round(result*100),"%")
    


# In[44]:


algoritmo(1,1,0,16,1)


# In[ ]:




