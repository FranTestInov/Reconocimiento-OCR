# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 20:26:29 2025
Modulo donde se crean todas las ventanas
@author: paco2
"""

import numpy as np
import cv2

def create_menu():
    menu_image = np.zeros((400, 1100, 3), np.uint8)
    #Despliegue del menu
    cv2.putText(menu_image, "Menu principal, seleccione una opcion o precione q para salir", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(menu_image, "1. Calibrar", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(menu_image, "2. Definir concentracion", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    #Mediciones
    cv2.putText(menu_image, "Mediciones", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(menu_image, "Concentracion de CO2 (GM-70): ", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_image, "Concentracion de CO2 (MH-Z19C): ", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_image, "Presion (BMP280): ", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_image, "Temperatura(DHT22): ", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_image, "Humedad(DHT22): ", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_image, "Bateria: ", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_image, "Bateria: ", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_image, "q. Salir", (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return menu_image

#Menu de concentraciones
def menu_concentraciones():
    menu_concentraciones_image = np.zeros((240, 400, 3), np.uint8)
    cv2.putText(menu_concentraciones_image, "Conecntraciones", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(menu_concentraciones_image, "1. CO2 0%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_concentraciones_image, "2. CO2 25% ", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_concentraciones_image, "3. CO2 50%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_concentraciones_image, "4. CO2 75%", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_concentraciones_image, "5. CO2 100%", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_concentraciones_image , "Precione q para salir: ", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Menu Concentraciones', menu_concentraciones_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyWindow('Menu Concentraciones')
            break
        
def menu_calibracion():
    menu_concentraciones_image = np.zeros((240, 400, 3), np.uint8)
    cv2.putText(menu_concentraciones_image, "Calibracion", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(menu_concentraciones_image, "1. Calibrar sensor de CO2", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu_concentraciones_image , "Precione q para salir: ", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Menu Calibracion', menu_concentraciones_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyWindow('Menu Calibracion')
            break
        
        
def crear_menu_mejorado(datos_sensores):
    """
    Crea una imagen de menú visualmente mejorada con layout dinámico y colores.

    Args:
        datos_sensores (dict): Un diccionario con los valores de los sensores.
                               Ej: {'co2_gm70': 3, 'co2_mhz19c': 450, ...}
    """
    # --- Configuración del Lienzo y Estilos ---
    ancho, alto = 800, 500
    menu_image = np.zeros((alto, ancho, 3), np.uint8)

    # Paleta de colores (en formato BGR de OpenCV)
    COLOR_FONDO = (0, 0, 0)
    COLOR_TITULO = (255, 255, 0)      # Cyan brillante
    COLOR_OPCION = (255, 255, 255)    # Blanco
    COLOR_ETIQUETA = (200, 200, 200)  # Gris claro
    COLOR_VALOR = (150, 255, 150)     # Verde claro
    COLOR_LINEA = (70, 70, 70)        # Gris oscuro

    # Fuentes
    font_principal = cv2.FONT_HERSHEY_SIMPLEX
    font_secundaria = cv2.FONT_HERSHEY_DUPLEX

    # --- Variables de Layout ---
    margen_x = 40
    columna_valores = 500  # Posición X donde empiezan los valores
    y_actual = 60
    altura_linea = 40

    # --- 1. Título Principal ---
    cv2.putText(menu_image, "Panel de Control", (margen_x, y_actual), font_principal, 1.2, COLOR_TITULO, 2)
    y_actual += 30
    cv2.line(menu_image, (margen_x, y_actual), (ancho - margen_x, y_actual), COLOR_LINEA, 1)
    y_actual += altura_linea

    # --- 2. Opciones del Menú ---
    cv2.putText(menu_image, "1. Calibrar Sensores", (margen_x, y_actual), font_secundaria, 0.9, COLOR_OPCION, 1)
    y_actual += altura_linea
    cv2.putText(menu_image, "2. Definir Concentracion", (margen_x, y_actual), font_secundaria, 0.9, COLOR_OPCION, 1)
    y_actual += int(altura_linea * 1.5) # Espacio extra antes de la siguiente sección

    # --- 3. Sección de Mediciones ---
    cv2.putText(menu_image, "Mediciones en Tiempo Real", (margen_x, y_actual), font_principal, 0.8, COLOR_TITULO, 2)
    y_actual += altura_linea

    # Lista de mediciones para mostrar
    mediciones = [
        ("Concentracion CO2 (GM-70)", datos_sensores.get('co2_gm70'), "ppm"),
        ("Concentracion CO2 (MH-Z19C)", datos_sensores.get('co2_mhz19c'), "ppm"),
        ("Presion (BMP280)", datos_sensores.get('presion'), "hPa"),
        ("Temperatura (DHT22)", datos_sensores.get('temperatura'), "°C"),
        ("Humedad (DHT22)", datos_sensores.get('humedad'), "%"),
        ("Nivel Bateria", datos_sensores.get('bateria'), "%")
    ]

    for etiqueta, valor, unidad in mediciones:
        # Si el valor no existe, muestra '---'
        valor_str = f"{valor} {unidad}" if valor is not None else "---"
        
        # Dibujar etiqueta a la izquierda
        cv2.putText(menu_image, f"{etiqueta}:", (margen_x + 20, y_actual), font_secundaria, 0.7, COLOR_ETIQUETA, 1)
        # Dibujar valor en la columna de valores
        cv2.putText(menu_image, valor_str, (columna_valores, y_actual), font_secundaria, 0.7, COLOR_VALOR, 1)
        y_actual += int(altura_linea * 0.8)

    # --- 4. Opción de Salida ---
    cv2.putText(menu_image, "q. Salir", (margen_x, alto - 40), font_principal, 0.8, COLOR_OPCION, 1)
    
    return menu_image

# --- EJEMPLO DE USO ---
if __name__ == '__main__':
    # Simula los datos que vendrían de tus sensores
    datos_actuales = {
        'co2_gm70': 3,
        'co2_mhz19c': 455,
        'presion': 1012.5,
        'temperatura': 24.8,
        'humedad': 65,
        'bateria': 87
    }

    # Crea la imagen del menú con los datos
    menu_visual = crear_menu_mejorado(datos_actuales)
    

    # Muestra la imagen
    cv2.imshow("Menu Mejorado", menu_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        