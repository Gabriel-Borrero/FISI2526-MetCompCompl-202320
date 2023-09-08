from mineral import Mineral
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math

class ExpansionTermicaMineral(Mineral):
    def __init__(self, nombre, dureza, rompimiento_por_fractura, color, composición, lustre, specific_gravity, sistema_cristalino):
        super().__init__(nombre, dureza, rompimiento_por_fractura, color, composición, lustre, specific_gravity, sistema_cristalino)
    
    #def __init__(self, archivo):
        #super().__init__()
        #self.archivo = archivo

    def abrir_archivo (self,archivo_csv):
        temperatura=[]
        volumen=[]
        ambos=[]
        archivo = open(archivo_csv, 'r', encoding='utf-8')
        linea=archivo.readline()
        linea=archivo.readline().strip()
        while linea!='':
            ambos.append(linea.split(","))
            linea=archivo.readline().strip()
        for i in range(0,len(ambos)):
            temperatura.append(float(ambos[i][0]))
            volumen.append(float(ambos[i][1]))

        return  temperatura, volumen 


    def coef_expansión_termica (self,archivo_csv):
        coeficientes=[]
        temperatura,volumen=self.abrir_archivo(archivo_csv)
        temperatura=np.array(temperatura)
        volumen=np.array(volumen)

        #Polinomio interpolador para calcular la derivada y luego usarla en la fórmula para calcular el coeficiente de expansión
        def interpolación (x=float):
            sum_ = volumen[0]
            Diff = np.zeros(( temperatura.shape[0],volumen.shape[0] ))
            h = temperatura[1]-temperatura[0]
            Diff[:,0] = volumen
            poly = 1.
            for i in range(1,len(temperatura)):
                poly *= (x-temperatura[i-1])
                for j in range(i,len(temperatura)):
                    Diff[j,i] = Diff[j,i-1] - Diff[j-1,i-1] 
                sum_ += poly*Diff[i,i]/(math.factorial(i)*h**(i))
                sum_=sym.expand(sum_)
            return sum_
        h = temperatura[1]-temperatura[0]
        d=0.
        if h!=0:
            for i in range(0,len(temperatura)):
                d=(interpolación((temperatura[i]+h))-interpolación((temperatura[i]-h)))/(2*h)
                coeficiente_a=(1/volumen[i])*d
                coeficientes.append(coeficiente_a)

        #Cálculo del error
        #Para que se puedan ver las gráficas, lo mejor es comentar el error puesto que python no lo está leyendo correctamente
        #error=np.std(coeficientes)/math.sqrt(len(temperatura))

        #Gráficas
        fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(18,4.5))
        axs[0].plot(temperatura,volumen)
        axs[0].set_ylabel(r'Volumen, (cc)')
        axs[0].set_xlabel('Temperatura (°C)')
        axs[0].set_title('Volumen vs Temperatura')

        axs[1].plot(temperatura,coeficientes)
        axs[1].set_ylabel(r'Coeficiente α')
        axs[1].set_xlabel('Temperatura (°C)')
        axs[1].set_title('Coeficiente vs Temperatura')

        plt.show()
        
        return coeficientes#, error

#mineral1 = ExpansionTermicaMineral('sulfur',2,True,'#7a785a','Mg3Si4O10(OH)2','NO METÁLICO',2.1,'ORTORRÓMBICO')
#print(mineral1.coef_expansión_termica('olivine_angel_2017.csv'))