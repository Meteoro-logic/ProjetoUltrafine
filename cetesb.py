# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:56:39 2020

@author: thela
"""
import numpy as np
import matplotlib.dates as dts
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from pylab import figure,arange
import pandas as pd
from datetime import datetime
import datacompy



def formata_dados(dados):
    df1 = pd.read_csv(dados,sep=';',encoding = 'unicode_escape',quotechar='"',decimal=".",  thousands=',',parse_dates=['Data'])
    variavel=df1.iloc[1]['Nome Parâmetro']
    unidade=df1.iloc[1]['Unidade Medida']
    df1 = df1.drop(["Tipo de Rede","Tipo de Monitoramento","Nome Parâmetro","Concentração","Taxa","Tipo","Dt. Instalação",
                    "Dt. Retirada","Código Estação","Média Móvel","Válido","Dt. Amostragem","Nome Estação","Unidade Medida"] ,axis=1)
    df1["Data"] = pd.to_datetime(df1["Data"])
    df1["Hora"] = [x + ':00' for x in df1["Hora"]]
    df1["Hora"] = pd.to_timedelta(df1["Hora"])
    df1["Data"] = df1["Data"] + df1["Hora"]
    df1 = df1.drop("Hora", axis=1)
    df1 = df1 .rename(columns = {"Média Horária": "Concentração de "+ variavel + " em " +unidade})
    composto =  ("Concentração de "+ variavel + " em " +unidade)
    return df1,composto


print("Favor definir a quantidade de variaveis.")
n = int(input())
listas = [[] for i in range(n)]
  
def get_arquivos(n):
    mylist=[]
    print("Favor definir o caminho e nome do arquivo com extenção (.csv, .txt, etc) seguido de enter")
    for i in range(n):
        print("Quantidade de variaveis salvas ate o momento:", i)
        dado = input()
        mylist.append(dado) 
    print (mylist)
    return mylist 

arquivos = get_arquivos(n)
n_linhas_arquivo = []
compostos =["Data"]



for i in range(n):
   listas[i], x = formata_dados(arquivos[i])
   compostos.append(x)
   n_linhas_arquivo.append(int(listas[i].shape[0]))
n_itera=min(n_linhas_arquivo)
matriz_final=pd.DataFrame({range(n+1)})
matriz_final.columns=compostos

for j in range(n_itera-1):
    for k in range(n_itera-1):
        if listas[0]["Data"][j]==listas[1]["Data"][k]:
           dta= listas[0]["Data"][k]
           var1=listas[0][compostos[n-1]][k]
           var2=listas[1][compostos[n]][k]
           matriz_final["Data"][j]=dta
           matriz_final[compostos[n-1]][j]=var1
           matriz_final[compostos[n]][j]=var2
           
           print(j,k)

                