import matplotlib.pyplot as plt

class Mineral:

    def __init__(self, nombre, dureza, rompimiento_por_fractura, color, composición, lustre, specific_gravity, sistema_cristalino):
        self.nombre = nombre
        self.dureza = dureza
        self.luste = lustre
        self.rompimiento_por_fractura = bool(rompimiento_por_fractura)
        self.color = color
        self.composición = composición
        self.sistema_cristalino = sistema_cristalino
        self.specific_gravity = float(specific_gravity)

    def es_silicato (self):
        if "Si" in self.composición and 'O' in self.composición:
            silicato=True
        else:
            silicato=False
        return silicato

    def densidad (self):
        densidad = float(1000*self.specific_gravity)
        return densidad

    def visualizar_color (self):
        fig,axs=plt.subplots(nrows=1,ncols=1,figsize=(1,1))
        axs.scatter(1,1,c=self.color,marker='o',s=500)
        plt.show()

    def dureza_rompimiento_atomos(self):
        print(f'Para el {self.nombre}, su dureza es {self.dureza}, su tipo de ropimiento es {self.rompimiento_por_fractura} y el sistema de organización de sus átomos es {self.sistema_cristalino}')
