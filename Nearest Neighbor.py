# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:40:54 2023

@author: Usuario
"""
import numpy as np;import matplotlib.pyplot as plt
import pandas as pd;from sklearn import metrics;import os;import random;from statistics import mode

k=[1,3,5]
#random.seed(1)
rutas_carpetas='C:/Users/Usuario/OneDrive/Documentos/Universidad/AAS/nearest neighbors/Datos/'
nombres_carpetas=os.listdir(rutas_carpetas)

def extraer_archivos():
    DATA=[]
    
    for carpeta in nombres_carpetas:
        archivos_texto=[];datos=[]
        ruta=rutas_carpetas+carpeta
        archivos=os.listdir(ruta)
        
        for archivo in archivos:
            if archivo[-4:]=='ures':
                archivos_texto.append(archivo)
                
        for texto in archivos_texto:
              lista_datos=[]
              ruta_nueva=ruta+'/'+texto
              
              dataframe=pd.read_csv(ruta_nueva,header=None,sep=':')
              dataframe=dataframe.drop([0,1,3,4,5],axis=0)
              DATAFRAME=pd.DataFrame(0,index=(),columns=['data'],dtype=object)
              
              lista=list(dataframe.loc[2,1])
              RAM=lista[0];contador=1
             
              while contador!=(len(lista)-1):
                  
                  if lista[contador]!=',':RAM=RAM+lista[contador]
                      
                  if lista[contador]==',':
                      RAM=float(RAM)
                      lista_datos.append(RAM)
                      contador+=1
                      RAM=lista[contador]
                  contador+=1
              
              for i in range(len(lista_datos)):
                  DATAFRAME.loc[i,'data']=lista_datos[i]
                  
              datos.append(DATAFRAME) 
        DATA.append(datos)  
        
    return DATA

def unificar_datos(data):
    lista_prueba=[];lista_entrenar=[];ram1=0;ram2=0
    
    for clases in range(len(data)):
        entrenar=pd.concat(data[clases][0:29])
        prueba=pd.concat(data[clases][30:39])
        lista_entrenar.append(entrenar)
        lista_prueba.append(prueba)
    
    indice_entrenar=np.arange(0,len(lista_entrenar[0]),1)
    indice_prueba=np.arange(0,len(lista_prueba[0]),1)
    
    entrenar_data=pd.DataFrame(0,index=indice_entrenar,columns=['colon-clear','esophagitis','ulcerative-colitis','extra'])
    prueba_data=pd.DataFrame(0,index=indice_prueba,columns=['colon-clear','esophagitis','ulcerative-colitis','extra'])
    
    for clases in range(len(lista_prueba)):
        for datos in range(prueba_data.shape[0]):
            x=lista_prueba[clases]
            prueba_data.iloc[datos,clases]=x.iloc[datos,0]

        for datos in range(entrenar_data.shape[0]):
            x=lista_entrenar[clases]
            entrenar_data.iloc[datos,clases]=x.iloc[datos,0]
  
    for i in range(len(lista_prueba[0])):
        valor=np.random.randint(0,2)
        prueba_data.loc[i,'extra']=valor
        
    for i in range(len(lista_entrenar[0])):
        valor=np.random.randint(0,2)
        entrenar_data.loc[i,'extra']=valor
        
    return entrenar_data,prueba_data

data=extraer_archivos()

entrenar_data,prueba_data=unificar_datos(data)

#posicion de archivos
#0 'colon-clear', 1'esophagitis', 2 'ulcerative-colitis'

def distancia_euclideana(entrenar_data,prueba_data,num_dato):
    distancia=np.zeros((entrenar_data.shape[0],2))
    P=prueba_data.drop('extra',axis=1);columnas=list(P.columns)
    
    for i in range(entrenar_data.shape[0]):
        D=0
        for m in columnas:
            D+=(prueba_data.loc[num_dato,m]-entrenar_data.loc[i,m])**2
        D=np.sqrt(D)
        distancia[i][0]=D
        distancia[i][1]=entrenar_data.loc[i,'extra']
        
    return distancia

def closest_neighbor(entrenar_data,prueba_data,num_dato,k):
    recomendado=[]
    distancia=distancia_euclideana(entrenar_data,prueba_data,num_dato)
    distancia=sorted(distancia,key=lambda x:x[0])
    
    for i in range(k):
        recomendado.append(distancia[i][1])
            
    vecino=mode(recomendado)

    return vecino

def algoritmo(k):
    precision=[];recall=[];Accuracy=[]
    data=extraer_archivos()

    entrenar_data,prueba_data=unificar_datos(data)
    
    tamano=prueba_data.shape[0]
    indice_prueba=np.arange(0,prueba_data.shape[0],1)
    resultado=pd.DataFrame(0,index=indice_prueba,columns=['colon-clear','esophagitis','ulcerative-colitis','extra','Guess1','Guess3','Guess5'])
   
    for clases in prueba_data.columns:
        for i in range(tamano):
            resultado.loc[i,clases]=prueba_data.loc[i,clases]
            
    for i in range(len(k)):
        x=str(k[i])
        palabra='Guess'+x
        for num_dato in range(tamano):
            vecino=closest_neighbor(entrenar_data, prueba_data, num_dato, k[i])
            resultado.loc[num_dato,palabra]=vecino

        matriz_confusion=metrics.confusion_matrix(resultado.loc[:,'extra'],resultado.loc[:,palabra])
        cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=matriz_confusion,display_labels=[False,True])
        
        accuracy=metrics.accuracy_score(resultado.loc[:,'extra'],resultado.loc[:,palabra])
        Precision=metrics.precision_score(resultado.loc[:,'extra'],resultado.loc[:,palabra])
        Recall=metrics.recall_score(resultado.loc[:,'extra'],resultado.loc[:,palabra])
        
        recall.append(Recall);Accuracy.append(accuracy);precision.append(Precision)
        
        cm_display.plot()
        plt.show()
        
    return resultado,recall,Accuracy,precision

resultado,recall,Accuracy,precision=algoritmo(k)
    
print('Los valores para k=1 son recall= ',recall[0],' accuracy fue de: ',Accuracy[0],' y precision: ',precision[0]);print();print()

print('Los valores para k=3 son recall= ',recall[1],' accuracy fue de: ',Accuracy[1],' y precision: ',precision[1]);print();print()

print('Los valores para k=5 son recall= ',recall[2],' accuracy fue de: ',Accuracy[2],' y precision: ',precision[2])
