import numpy as np
def posicion(A,k,m,t):
    '''
    Esta función retorna la posición, x, en un tiempo de de una masa m atada a un
    resorte de constante de fuerza k que inicialmente es desplazada A unidades de
    su posición de equilibrio. Esta función devuelve el resultado de 

    $A\sin (\omega t)$ donde $\omega = \sqrt{\frac{k}{m}}$

    Parámetros
    ----------
    A: float
        La amplitud o desplazamiento inicial de la masa.
    k: float
        La constante de fuerza del resorte.
    m: float
        La masa del objeto.
    t: float
        El tiempo en el cual se quiere calcular la posición.
    '''
    omega = np.sqrt(k/m)
    return A*np.sin(omega*t)


def DerivadaDerecha(f,x,h):
    
    d = 0.
    
    if h != 0:
        d = (f(x+h) - f(x))/h
        
    return d
def DerivadaIzquierda(f,x,h):
    
    d = 0.
    
    if h != 0:
        d = (f(x) - f(x-h))/h
        
    return d

def DerivadaCentral(f,x,h):
    
    d = 0.
    
    if h != 0:
        d = (f(x+h) - f(x-h))/(2*h)
        
    return d

 list=[]
        for i in contenido:
            list.append(i)
            
        for i in list:
            if i == '    data: |\n':
                x=list.index(i)
                del list[0:x+1]
                
                
#definitiva

   while final==False:    
            for i in y:
                if i == ['data: |']:   
                    x=y.index(i)
                    del y[0:x+1]
            final=True