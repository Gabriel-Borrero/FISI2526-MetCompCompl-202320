#Punto 1.3.

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
                    print(x)
                    del y[x-1:x+1]
                


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

print(tuplas_archivos_yml('C:\\Users\\gaboe\\Desktop\\FISI2526-MetCompCompl-202320\\Taller_1\\Plásticos_Comerciales\\French.yml'))
print(tuplas_archivos_yml)