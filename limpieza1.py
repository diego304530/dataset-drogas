#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

datos = pd.read_csv("encuestaFinal.csv",  sep=",")

headers= ['estrato','genero','edad', 'grado', 'vive con', 'estado conyugal padres', 'zona','deporte' , 'trabaja',
           'invitacion a consumir drogas','consecuencias consumo padres', 'habito-padres-alcohol','reaccion padres','cigarrillo','consume droga',
          'a que edad', 'en que lugar', 'atencion de padres', 'conocimiento de amistades padres', 'continua', 'grados perdidos', 
          'recreo', 'acoso escolar', 'acoso sexual', 'relacion familiar', 'limitaciones', 'sexualidad', 'personalidad', 'habitacion compartida',
          'perdida de familiar', 'rupturas amorosas', 'situacion sentimental', 'otro colegio', 'familiares consumidores', 'colegio']
datos.columns = headers

datos['colegio']=datos['colegio'].fillna('san francisco')


# In[7]:


datos.head(5)


# In[ ]:





# In[ ]:





# In[8]:


datos["genero"].replace("Prefiero no decirlo", "Hombre", inplace =True)
datos["vive con"].replace("Padres y hermanos", "Solo Padre" , inplace=True)
datos["vive con"].replace("madre y padre ", "Madre y Padre" , inplace=True)
datos["vive con"].replace("padre y hermano ", "Solo Padre" , inplace=True)
datos["sexualidad"].replace("Bisexual", "Heterosexual" , inplace=True)

datos["personalidad"].replace("Tímido", "Timido" , inplace=True)  
datos["habito-padres-alcohol"].replace("Nunca toma alcohol", "Nunca toman alcohol" , inplace=True)
datos["atencion de padres"].replace("Nunca o casi nunca saben donde estoy", "Nunca o casi nunca saben dónde estoy" , inplace=True)
datos["atencion de padres"].replace("Siempre o casi siempre saben dónde estoy", "Siempre o casi siempre saben dónde estoy" , inplace=True)

datos["en que lugar"].replace("casa", "Casa" , inplace=True)
datos["en que lugar"].replace("calle", "Calle" , inplace=True)
datos["habitacion compartida"].replace("A veces compartes habitación","Si",inplace=True)

datos["sexualidad"].replace("Heterosexual ", "Heterosexual" , inplace=True)


# In[9]:



vive= pd.get_dummies(datos['vive con'])
datos= pd.concat([datos, vive], axis=1)
datos.drop(["vive con"], axis=1, inplace=True)
conyugal= pd.get_dummies(datos['estado conyugal padres'])
datos= pd.concat([datos, conyugal], axis=1)
datos.drop(["estado conyugal padres"], axis=1, inplace=True)
datos["a que edad"] = datos["a que edad"].fillna(0)

datos["continua"].replace("Nunca he consumido drogas", 0 , inplace=True)


# In[10]:



for i in datos:
    datos[i].replace("Si", 1 , inplace=True)
    datos[i].replace("No", 0 , inplace=True)
        
datos["genero"].replace("Mujer", 0 , inplace=True)
datos["genero"].replace("Hombre", 1 , inplace=True)
datos["zona"].replace("Urbana", 1 , inplace=True)
datos["zona"].replace("Rural", 0 , inplace=True)

datos["situacion sentimental"].replace("Soltero/a", 1 , inplace=True)
datos["situacion sentimental"].replace("Novio/a", 0 , inplace=True)

# categorizacion reaccion de los padres
datos["reaccion padres"].replace("Indiferentes", 1 , inplace=True)
datos["reaccion padres"].replace("Poco molestos", 2 , inplace=True)
datos["reaccion padres"].replace("Bastante molestos", 3 , inplace=True)
datos["reaccion padres"].replace("Extremadamente molestos", 4 , inplace=True)

# categorizacion de atencion de padres
datos["atencion de padres"].replace("Nunca o casi nunca saben dónde estoy", 1 , inplace=True)
datos["atencion de padres"].replace("A veces no saben", 2 , inplace=True)
datos["atencion de padres"].replace("Siempre o casi siempre saben dónde estoy", 3 , inplace=True)
# habitos alcohol de consumo de los padres Nunca toma alcohol

datos["habito-padres-alcohol"].replace("Nunca toman alcohol", 1 , inplace=True)
datos["habito-padres-alcohol"].replace("Solo en algunas ocaciones", 2 , inplace=True)
datos["habito-padres-alcohol"].replace("Al menos un trago semanalmente", 3 , inplace=True)
# categorizacion de conocimiento de amistades de los hijos 
datos["conocimiento de amistades padres"].replace("Poco", 1 , inplace=True)
datos["conocimiento de amistades padres"].replace("Mas o menos", 2 , inplace=True)
datos["conocimiento de amistades padres"].replace("Bastante", 3 , inplace=True)

datos["sexualidad"].replace("Homosexual", 0 , inplace=True)
datos["sexualidad"].replace("Heterosexual", 1 , inplace=True)

datos["colegio"].replace("san francisco", 1 , inplace=True)
datos["colegio"].replace("Carlos M simmonds", 2 , inplace=True)
datos["colegio"].replace("Rafael Pombo", 3 , inplace=True)

lugar = pd.get_dummies(datos['en que lugar'])
datos= pd.concat([datos, lugar], axis=1)
datos.drop(["en que lugar"], axis=1, inplace=True)

personalidad = pd.get_dummies(datos['personalidad'])
datos= pd.concat([datos, personalidad], axis=1)
datos.drop(["personalidad"], axis=1, inplace=True)


# In[ ]:





# In[ ]:





# In[11]:


datos.head(5)


# In[ ]:





# In[15]:


headers1=['Estrato','Genero','Edad','Grado','Zona','Deporte','Trabaja','Invitacion_drogas','Consecuencias_consumo',
         'Padres_alcohol','Reaccion_padres','Cigarrillo','Consume_droga','A_que_edad','Atencion_padres',
         'Conocimiento_amistades_padres','Continua','Grados_perdidos','Recreo','Acoso_escolar','Acoso_sexual',
         'Relacion_familiar','Limitaciones','Sexualidad','Habitacion_compartida','Perdida_familiar',
         'Rupturas_amorosas','Situacion_sentimental','Otro_colegio','Familiares_consumidores',
         'Colegio','MadreyPadre','Otros_familiares','Solo Madre','Solo Padre','Casados','Separados',
         'Soltero_soltera','Unión_libre','Viudo_viuda','Agresivo','Alegre','Extrovertido','Malhumorado',
         'Timido','Impulsivo','Calle','Casa','Colegio','Con_pareja_sentimental','Fiestas','No_consumido',
         'Parque' ,'Paseo']


# In[16]:


datos.columns = headers1


# In[12]:


datos.head(5)


# In[19]:


datos.to_csv("encuestaBinaria.csv")


# In[ ]:




