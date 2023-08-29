def tuplas_archivos_yml(ruta_de_archivo):
    with open(ruta_de_archivo, 'r') as archivo:
        contenido = archivo.readlines()
   
        
    return contenido

print(tuplas_archivos_yml('C:\\Users\\gaboe\\Desktop\\FISI2526-MetCompCompl-202320\\Taller_1\\Plásticos_Comerciales\\French.yml'))
print(tuplas_archivos_yml)