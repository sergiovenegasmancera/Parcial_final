import cv2
import numpy as np
import os
import sys
from Bandera import *

if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)

    image = cv2.imread(path_file)
    flag = Bandera(image)
    num_colores = flag.colores()
    print("Numero de colores: ", num_colores)
    porcentaje = flag.porcentaje()
    print("Porcentaje de cada color presente en la bandera: ",porcentaje)
    orientacion = flag.orientacion()
    print("Orientación de la bandera: ",orientacion)

