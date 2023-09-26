import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math

#Punto 1 - A (Gráfico de la función)
B0 = 0.05  # Tesla
f = 7  # Hz
Omega = 3.5  # rad/s
R = 1750.0  # Resistencia en Ohms (1.75 kΩ)
r = 0.25  # Radio del bucle de cobre en metros

# Función para calcular el flujo magnético ΦB en función del tiempo t
def calcular_flujo_magnetico(t):
    return np.pi * r**2 * B0 * np.cos(Omega * t) * np.cos(2 * np.pi * f * t)

# Función para calcular la corriente inducida en función del tiempo t
def calcular_corriente(t):
    phi_B = devcentral(calcular_flujo_magnetico, t, h)
    return -phi_B/ R

# Periodo de rotación del bucle
T_rotation = 2 * np.pi / Omega

t = np.linspace(0, 2 * T_rotation, 1000)
y=calcular_flujo_magnetico(t)

def devcentral(f,x,h):
    d=0.
    if h!=0:
        d=(f(x+h)-f(x-h))/(2*h) #la formula es asi, con el 2, es central porque analizamos la derivada del punto que esta en el centro de adelante y atras.
        return d
h=t[1]-t[0]
Cdev=devcentral(calcular_flujo_magnetico, t, h)
I = calcular_corriente(t)

# Grafica la corriente inducida
plt.figure(figsize=(10, 6))
plt.plot(t, I)
plt.xlabel('Tiempo (s)')
plt.ylabel('Corriente Inducida (A)')
plt.title('Corriente Inducida en el Bucle en función del Tiempo')
plt.grid(True)
plt.show()

#Punto 1-B (Los 3 primeros timepos/Raíces)

def Funcion(x):
    return (np.pi*r**2)*(B0)*np.cos(Omega*x)*np.cos(2*np.pi*f*x)

x = np.linspace(0,0.0015,100)
y = Funcion(x)

#Hallar derivada
def derivada(funcion, x, h=1e-10):
    d=0.
    if h!=0:
        d=(funcion(x+h)-funcion(x-h))/(2*h) 
        return d 
def NewtonRaphson(xi):
    #parametros Iniciales
    fxi = Funcion(xi)
    dxi = derivada(Funcion, xi)
    for i in range(50): 
        x = xi - (fxi/dxi)
        fxi = Funcion(x)
        dxi = derivada(Funcion, x) 
        xi = x 
    return x

def GetAllRoots(x,Funcion):
    Roots = []
    for i in x:
        root = NewtonRaphson(i)
        root = np.round(root,10) 
        Roots.append(root)
        
    unique_roots = [] 
    for i in Roots:

        if i not in unique_roots and i>=0:

            unique_roots.append(i)
            
    unique_roots.sort()   
    return unique_roots
 

print("para el punto 1.2, las primeras 3 raices (los primeros tres instantes de tiempo en los que la corriente sobre el bucle es cero) son: ", GetAllRoots(x,Funcion)[:3])

"----------------------------------------------------------------------------------------------------------------------------------"

#2 - Este punto se encuentra como imagen (jpg) en el repositorio. 

"----------------------------------------------------------------------------------------------------------------------------------"

#3.1 - Gauss-Laguerre: 

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)


def GetLaguerreRecursive(n,x):

    if n==0:
        poly = sym.Number(1)
    elif n==1:
        poly = 1 - x
    else:
        poly = ((2*(n-1)+1-x)*GetLaguerreRecursive(n-1,x)-(n-1)*GetLaguerreRecursive(n-2,x))/n
   
    return sym.expand(poly,x)

def GetDLaguerre(n,x):
    Pn = GetLaguerreRecursive(n,x)
    return sym.diff(Pn,x,1)

def GetNewton(f,df,xn,itmax=10000,precision=1e-14):
    
    error = 1.
    it = 0
    
    while error >= precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/df(xn)
            
            error = np.abs(f(xn)/df(xn))
            
        except ZeroDivisionError:
            print('Zero Division')
            
        xn = xn1
        it += 1
        
    if it == itmax:
        return False
    else:
        return xn
    
def GetRoots(f,df,x,tolerancia = 10):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)

        if  type(root)!=bool:
            croot = np.round( root, tolerancia )
            
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots


def GetAllRootsGLag(n):

    xn = np.linspace(0,n + (n-1)*math.sqrt(n), 100)
    
    Laguerre = []
    DLaguerre = []
    
    for i in range(n+1):
        Laguerre.append(GetLaguerreRecursive(i,x))
        DLaguerre.append(GetDLaguerre(i,x))
    
    poly = sym.lambdify([x],Laguerre[n],'numpy')
    Dpoly = sym.lambdify([x],DLaguerre[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')
    
    return Roots


def GetWeightsGLag(n):

    Roots = GetAllRootsGLag(n)

    

    Laguerre = []
    
    for i in range(n+2):
        Laguerre.append(GetLaguerreRecursive(i,x))
    
    poly = sym.lambdify([x],Laguerre[n+1],'numpy')
    Weights = Roots/(((n+1)**2)*(poly(Roots))**2)
    
    return Weights

n = 5
funcion = lambda x: -2*np.sqrt(x/np.pi)
raices = GetAllRootsGLag(n)
pesos = GetWeightsGLag(n)
I = 0
for i in range(n):
    I += pesos[i]*funcion(raices[i])


"----------------------------------------------------------------------------------------------------------------------------------"

#3.2 - Gauss-Hermite: 


def GetHermite(n,x):

    if n==0:
        poly = sym.Number(1)
    elif n==1:
        poly = 2*x 
    else:
        poly = (2*x)*GetHermite(n-1,x) - (2*(n-1))*GetHermite(n-2,x)
    
    return sym.expand(poly,x)




def GetDHermite(n,x):
    Pn = GetHermite(n,x)
    return sym.diff(Pn,x,1)

def GetNewton(f,df,xn,itmax=10000,precision=1e-14):
    
    error = 1.
    it = 0
    
    while error >= precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/df(xn)
            
            error = np.abs(f(xn)/df(xn))
            
        except ZeroDivisionError:
            print('Zero Division')
            
        xn = xn1
        it += 1
        
    if it == itmax:
        return False
    else:
        return xn
    


def GetAllRootsGHer(n):

    xn = np.linspace(-np.sqrt((4*n+1)),np.sqrt((4*n+1)),100)
    
    Hermite = []
    DHermite = []
    
    for i in range(n+1):
        Hermite.append(GetHermite(i,x))
        DHermite.append(GetDHermite(i,x))
    
    poly = sym.lambdify([x],Hermite[n],'numpy')
    Dpoly = sym.lambdify([x],DHermite[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')
    
    return Roots


def GetWeightsGHer(n):

    Roots = GetAllRootsGHer(n)
    Hermite= []
    
    for i in range(n):
        Hermite.append(GetHermite(i,x))
    
    poly = sym.lambdify([x],Hermite[n-1],'numpy')
    Weights = (2*(n-1) * math.factorial(n) * np.sqrt(np.pi))/(n*2 *(poly(Roots))**2)
    
    return Weights

n = 3
raices = GetAllRootsGHer(n)
pesos = GetWeightsGHer(n)


funcion = lambda x: x**4/(math.e**(-x)**2) #Esta la cambias por la función que quieras 

I = 0
for i in range(n):
    I += pesos[i]*funcion(raices[i])
    

"----------------------------------------------------------------------------------------------------------------------------------"


#3.3. Aplicación:

"----------------------------------------------------------------------------------------------------------------------------------"
#1 Realizando la sustitución...(la demostración teórica de este punto se encuentra como imagen (jpg) en el repositorio). Por otro lado, el código para comprobar que la integral es igual a 1 es:

n = 5
raices = GetAllRootsGLag(n)
pesos = GetWeightsGLag(n)
funcion = lambda x: 2*np.sqrt(x/np.pi)#Esta la cambias por la función que quieras 

I = 0
for i in range(n):
    I += pesos[i]*funcion(raices[i])
print("La integral da un valor de: " +str(I)+ " demostrando que sí es una distribución de probabilidad")

"----------------------------------------------------------------------------------------------------------------------------------"
 
 #2 Grafique P(v) para distintas temperaturas. ¿Qué puede decir de la velocidad más probable (con mayor P(v)) a medida que aumenta la temperatura?:
 
x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)

def GetLaguerreRecursive(n,x):

    if n==0:
        poly = sym.Number(1)
    elif n==1:
        poly = 1 - x
    else:
        poly = ((2*(n-1)+1-x)*GetLaguerreRecursive(n-1,x)-(n-1)*GetLaguerreRecursive(n-2,x))/n

    return sym.expand(poly,x)

def GetDLaguerre(n,x):
    Pn = GetLaguerreRecursive(n,x)
    return sym.diff(Pn,x,1)

def GetNewton(f,df,xn,itmax=10000,precision=1e-14):

    error = 1.
    it = 0

    while error >= precision and it < itmax:

        try:

            xn1 = xn - f(xn)/df(xn)

            error = np.abs(f(xn)/df(xn))

        except ZeroDivisionError:
            print('Zero Division')

        xn = xn1
        it += 1

    if it == itmax:
        return False
    else:
        return xn

def GetRoots(f,df,x,tolerancia = 10):

    Roots = np.array([])

    for i in x:

        root = GetNewton(f,df,i)

        if  type(root)!=bool:
            croot = np.round( root, tolerancia )

            if croot not in Roots:
                Roots = np.append(Roots, croot)

    Roots.sort()

    return Roots


def GetAllRootsGLag(n):

    xn = np.linspace(0,n + (n-1)*math.sqrt(n), 100)

    Laguerre = []
    DLaguerre = []

    for i in range(n+1):
        Laguerre.append(GetLaguerreRecursive(i,x))
        DLaguerre.append(GetDLaguerre(i,x))

    poly = sym.lambdify([x],Laguerre[n],'numpy')
    Dpoly = sym.lambdify([x],DLaguerre[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)

    if len(Roots) != n:
        ValueError('El número de raíces debe ser igual al n del polinomio.')

    return Roots


def GetWeightsGLag(n):

    Roots = GetAllRootsGLag(n)



    Laguerre = []

    for i in range(n+2):
        Laguerre.append(GetLaguerreRecursive(i,x))

    poly = sym.lambdify([x],Laguerre[n+1],'numpy')
    Weights = Roots/(((n+1)**2)*(poly(Roots))**2)

    return Weights

n = 5
raices = GetAllRootsGLag(n)
pesos = GetWeightsGLag(n)


#3.2

M=1
R=8.31
v= np.linspace(0, 50, 50)

plt.figure()
for T in v:
    p=4*np.pi*(M/(2*np.pi*R*T))**(3/2)*(v**2)*math.e**(-(M*v**2)/(2*R*T))
    plt.plot(v,p)

plt.xlabel('Velocidad (v)')
plt.ylabel('Probabilidad de velocidad (P(v))')
plt.title('Probabilidad vs Velocidad')
plt.show()
 
 
 #3 Para 10 distintas temperaturas, encuentre la velocidad promedio ...
 
M=1
R=8.31
T = np.linspace(0,10,10)

for i in T:
    funcion_avg = lambda u: 2*np.sqrt((2*u*R*T)/M)*np.sqrt(u/np.pi) #Esta la cambias por la función que quieras
    I_avg = 0
    for i in range(n):
        I_avg += pesos[i]*funcion_avg(raices[i])
    vr_avg=np.sqrt((8*R*T)/(np.pi*M))
    
print("Para el punto 3, subíndice 3, tenemos lo siguiente: ")
print(f"Los valores que nos saca la integral son: {I_avg}")
print(f"Los valores teóricos son: {vr_avg}")

plt.figure()
plt.loglog(T,I_avg)
plt.xlabel('Temperatura')
plt.ylabel('Velocidad Promedio') 
plt.title('Temperatura vs Velocidad Promedio')
plt.show()

 #4 Para 10 distintas temperaturas, encuentre la velocidad media cuadrática...
 
 
for i in T:
    funcion_rms = lambda u: 2*((2*u*R*T)/M)*np.sqrt(u/np.pi)
    I_rms = 0
    for i in range(n):
        I_rms += pesos[i]*funcion_rms(raices[i])
    I_rms=np.sqrt(I_rms)
    vr_rms=np.sqrt((3*R*T)/M)
    
print("Para el punto 3, subíndice 4, tenemos lo siguiente: ")
print(f"Los valores que nos saca la integral son: {I_rms}")
print(f"Los valores teóricos son: {vr_rms}")

plt.figure()
plt.loglog(T,I_rms) 
plt.xlabel('Temperatura')
plt.ylabel('Velocidad Media Cuadrática') 
plt.title('Temperatura vs Velocidad Media Cuadrática')
plt.show()
 
 
 #5 Usando lo anterior, demuestre que la energ´ıa interna de un gas... (Este punto se encuentra como imagen (jpg) en el repositorio).
 
 
 
"----------------------------------------------------------------------------------------------------------------------------------"
 #4 - Método de Montecarlo.
"----------------------------------------------------------------------------------------------------------------------------------"
  #1 - Grafique el error porcentual para la integral ...:
 

N_v=[]
for i in np.linspace(10, 10**5, 100):
    N_v.append(i)
    
a = 0
b = np.pi
y=[]
for i in range(len(N_v)):
    x = np.random.uniform(a,b,int(N_v[i]))
    
    def func_integrate(x):
        
        return np.exp(-x)*np.sin(x)


    fi = func_integrate(x)

    I = (b-a)*sum(fi)/N_v[i]


    Iteo = 0.5*(1+np.exp(-np.pi))
    error=np.abs(1-I/Iteo)
    y.append(error)



fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(18,7))
axs.set_ylabel(r'Error porcentual')
axs.set_xlabel('número de muestras N')
axs.set_title("Error porcentual vs número de muestras N")
axs.scatter(N_v, y)
axs.plot(N_v, y)
plt.show()

"----------------------------------------------------------------------------------------------------------------------------------"
#3  La ecuación del transporte de neutrones requiere calcular la tasa de producci´on de neutrones [1] por medio de la integral...:
"----------------------------------------------------------------------------------------------------------------------------------" 

R = 1
N = 100000
I=0
x = np.random.uniform(-R,R,N)
y = np.random.uniform(-R,R,N)
z = np.random.uniform(-R,R,N)

suma = 0

for i in range(N):
    if (x[i]**2+y[i]**2+z[i]**2)<=1:
        suma += np.sin(x[i]**2+y[i]**2+z[i]**2)*math.e**(x[i]**2+y[i]**2+z[i]**2)
I=(8*suma)/N
print("El valor aproximado de la integral mediante el método de Montecarlo es: "+str(I))

