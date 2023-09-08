#Punto 1.3.

import matplotlib.pyplot as plt
import numpy as np
def tuplas_archivos_yml(ruta_de_archivo):
    with open(ruta_de_archivo, 'r') as archivo:
        contenido = archivo.readlines()
        list=[]
        y=[]
        
        tuplas=[]
        
        for i in contenido:
            list.append(i)
            
        for i in range(len((list))):
                y.append(list[i].strip().split(","))
            
        del y[0:9] 
          
        for i in y:
            for h in i:
           
                if h =='data: |':   
                    x=y.index(i)
                    del y[x-1:x+1]
                    
        for i in y:
            for h in i:
           
                if h =='SPECS:':   
                    x=y.index(i)
                    y=y[:x]
        
    for listas in y:
        for cadena in listas:
            valores = cadena.split(" ") 
            if len(valores) == 2:
                try:
                 valor1 = float(valores[0]) 
                 valor2 = float(valores[1])  
                 tupla = (valor1, valor2)    
                 tuplas.append(tupla)
                except ValueError:
                    print(f"Error al convertir valores en la cadena: {cadena}")              
    return tuplas

#print(tuplas_archivos_yml('C:\\Users\\gaboe\\Desktop\\FISI2526-MetCompCompl-202320\\Taller_1\\Adhesivos_Ópticos\\Iezzi.yml.1'))



#Punto 1.4.

def graficos(funcion, ruta_1, ruta_2):
    promedio_n=0
    promedio_na=0
    
    #Gráfico para el Krapton:
    
    
    funcion=tuplas_archivos_yml(ruta_1)
    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(18,4.5))
    for i in range(len(funcion)):
            
            x=funcion[i][0]
            y=funcion[i][1]
            axs.scatter(x,y)
            promedio_n+=y
            
    promedio_n=promedio_n/i 
   
    valores_y = [tupla[1] for tupla in funcion]  
    desviacion_estandar = np.std(valores_y)
  
    axs.set_ylabel(r'Índice de refracción n(i)')
    axs.set_xlabel('Longitud de onda λ(i)')
    axs.set_title('Longitud de onda λ(i) vs Índice de refracción n(i). \nPromedio ni: ' + str(promedio_n)+ " \nDesviación Estandar: " + str(desviacion_estandar))
       
    plt.show()
            
    #Gráfico para NOA138:
    i=0
    funcion=tuplas_archivos_yml(ruta_2)
    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(18,4.5))
    for i in range(len(funcion)):
            
            x=funcion[i][0]
            ya=funcion[i][1]
            axs.scatter(x,ya)
            promedio_na+=ya
            
    promedio_na=promedio_na/i 

    valores_ya = [tupla[1] for tupla in funcion]  
    desviacion_estandara = np.std(valores_ya)
  
    axs.set_ylabel(r'Índice de refracción n(i)')
    axs.set_xlabel('Longitud de onda λ(i)')
    axs.set_title('Longitud de onda λ(i) vs Índice de refracción n(i). \nPromedio ni: ' + str(promedio_na)+ " \nDesviación Estandar: " + str(desviacion_estandara))
    plt.show()
    
    return None

ruta_1='C:\\Users\\gaboe\\Desktop\\FISI2526-MetCompCompl-202320\\Taller_1\\Plásticos_Comerciales\\French.yml'
ruta_2="C:\\Users\\gaboe\\Desktop\\FISI2526-MetCompCompl-202320\\Taller_1\\Adhesivos_Ópticos\\Iezzi.yml.1"
#print(graficos(tuplas_archivos_yml,ruta_1,ruta_2))



#punto 1.5.

def general_graficos(funcion, ruta_de_archivo):
    
    
    with open(ruta_de_archivo, 'r') as archivo:
     contenido = archivo.readlines()
     
     for i in contenido[1:]:
            ruta_primordial="C:\\Users\\gaboe\\Desktop\\FISI2526-MetCompCompl-202320\\Taller_1\\"
            conector="\\"
            
                
            material=i.split(",")
            material=material[0]
            
            
            nombre_2=i.split(",")
            nombre_2=nombre_2[2]
            if nombre_2=="Materia InorgÃ¡nica":
                nombre_2="Materia_Inorgánica"
            
            try: 
                
                ruta_específica=ruta_primordial+material+conector+nombre_2+".yml"
                funcion=tuplas_archivos_yml(ruta_específica)
            except:
                print("cargando")
            
            promedio_n=0
            valores_y = [tupla[1] for tupla in funcion]
            valores_x = [tupla[0] for tupla in funcion]
            
    
            desviacion_estandar = np.std(valores_y)
            fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(18,7))
            promedio_na=0
            for i in range(len(funcion)):
            
                ya=funcion[i][1]
                promedio_na+=ya
            
            promedio_na=promedio_na/i 
            axs.scatter(valores_x, valores_y)
            axs.plot(valores_x, valores_y)
            axs.set_ylabel(r'Índice de refracción n(i)')
            axs.set_xlabel('Longitud de onda λ(i)')
            axs.set_title('Longitud de onda λ(i) vs Índice de refracción n(i). \nPromedio ni: ' + str(promedio_na)+ " \nDesviación Estandar: " + str(desviacion_estandar))
            ruta_específica=ruta_primordial+material+conector+nombre_2+".png"
            plt.savefig(ruta_específica)
            
            plt.close()
           
        
            
    return None
    
print(general_graficos(tuplas_archivos_yml, "C:\\Users\\gaboe\\Desktop\\FISI2526-MetCompCompl-202320\\Taller_1\\indices_refraccion.csv"))
        
        
