import matplotlib.pyplot as plt
import numpy as np
from mineral import Mineral

minerales_arreglo=[]
archivo = open('minerales.txt', "r", encoding='utf-8')
linea=archivo.readline().strip()
i=0
for x in archivo:
    minerales_arreglo.append(x)
while i in range (0,17):
    minerales_arreglo[i]=minerales_arreglo[i].split('\t')
    i+=1

print(minerales_arreglo)

def numero_de_silicatos (minerales_arreglo):
    count=0
    j=0
    while j <= 16:
        mineral=Mineral(minerales_arreglo[j][0], minerales_arreglo[j][1],minerales_arreglo[j][2],minerales_arreglo[j][3],minerales_arreglo[j][4],minerales_arreglo[j][5],minerales_arreglo[j][6],minerales_arreglo[j][7])
        if mineral.es_silicato() == True:
            count+=1
        j+=1
    return count

print(numero_de_silicatos(minerales_arreglo))

def densidad_promedio (minerales_arreglo):
    densidades=0
    i=0
    while i <= 16:
        mineral=Mineral(minerales_arreglo[i][0], minerales_arreglo[i][1],minerales_arreglo[i][2],minerales_arreglo[i][3],minerales_arreglo[i][4],minerales_arreglo[i][5],minerales_arreglo[i][6],minerales_arreglo[i][7])
        densidades+=mineral.densidad()
        i+=1
    promedio_densidades=densidades/17
    return promedio_densidades

print(densidad_promedio(minerales_arreglo))
        