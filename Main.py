# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 20:10:10 2025
Versión del calibrador automático utilizando Tesseract OCR.
@author: paco2
"""

# 1. IMPORTS
import cv2
import numpy as np
import os
import pytesseract
import serial
import time
from collections import Counter
 

# 2. CONFIGURACIÓN
# Esta línea obtiene la ruta absoluta de la CARPETA donde se encuentra este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    """Valores iniciales y de configuración para la aplicación."""
    WINDOW_NAME_CAM = 'Camara'
    WINDOW_NAME_MENU = 'Menu'
    
    SERIAL_PORT = 'COM3'
    BAUD_RATE = 115200
    
    # --- ¡MUY IMPORTANTE! ---
    # Pytesseract necesita saber dónde está el programa Tesseract.exe.
    # Cambiá esta ruta para que apunte a donde lo instalaste en tu PC.
    # La 'r' al principio es para que Python interprete la cadena de texto correctamente.
    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Valores iniciales para la ROI grande de detección
    BIG_ROI_INITIAL_X = 150
    BIG_ROI_INITIAL_Y = 240
    BIG_ROI_INITIAL_W = 200
    BIG_ROI_INITIAL_H = 120
    # Tamaño del búfer: cuántas lecturas válidas recientes vamos a almacenar.
    READING_BUFFER_SIZE = 15
    # Umbral de confianza: qué porcentaje de las lecturas en el búfer deben ser
    # iguales para considerar el valor como "estable". (0.6 = 60%)
    CONFIDENCE_THRESHOLD = 0.6

# 3. CLASE PRINCIPAL DE LA APLICACIÓN
class CalibratorApp:
    """
    Clase principal que encapsula toda la lógica de la aplicación de calibración.
    """
    def __init__(self, config):
        """
        Constructor: Inicializa Tesseract, la cámara y la interfaz de usuario.
        """
        self.config = config
        
        # --- Configuración de Tesseract ---
        # Le decimos a pytesseract dónde encontrar el ejecutable.
        # Esto se hace una sola vez al iniciar la aplicación.
        try:
            pytesseract.pytesseract.tesseract_cmd = self.config.TESSERACT_CMD
            # Hacemos una llamada de prueba para verificar que Tesseract funciona
            print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print("--- ERROR DE TESSERACT ---")
            print(f"No se pudo encontrar o ejecutar Tesseract en la ruta: {self.config.TESSERACT_CMD}")
            print("Asegurate de que la ruta en la clase Config sea correcta.")
            print(f"Error original: {e}")
            print("--------------------------")
            # Podríamos decidir salir si Tesseract no funciona
            # return

        self.ser = None # Inicializamos como None
        try:
            # Intentamos abrir el puerto serial definido en la configuración
            self.ser = serial.Serial(self.config.SERIAL_PORT, self.config.BAUD_RATE, timeout=1)
            time.sleep(2) # Damos tiempo al puerto para que se establezca la conexión
            print(f"Puerto serial {self.config.SERIAL_PORT} conectado exitosamente.")
        except serial.SerialException as e:
            print("--- ERROR SERIAL ---")
            print(f"No se pudo abrir el puerto {self.config.SERIAL_PORT}. ¿Está conectado el ESP32?")
            print(f"Error original: {e}")
            print("La aplicación continuará sin comunicación serial.")
            print("--------------------")
            
        # Abre la cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("No se puede abrir la cámara")
            
        # Atributos de estado de la ROI grande
        self.big_roi_x = self.config.BIG_ROI_INITIAL_X
        self.big_roi_y = self.config.BIG_ROI_INITIAL_Y
        self.big_roi_w = self.config.BIG_ROI_INITIAL_W
        self.big_roi_h = self.config.BIG_ROI_INITIAL_H
        
        # Threshold - Valor default
        self.threshold_value = 150
        
        # Un búfer (lista) para almacenar las últimas lecturas válidas.
        self.readings_buffer = []
        # El último valor que consideramos estable y que se muestra en el menú.
        self.stable_reading = "---" # Valor inicial
        
        # Inicialización de la UI
        self._setup_ui()

    def _setup_ui(self):
        """Configura las ventanas y los controles de OpenCV."""
        cv2.namedWindow(self.config.WINDOW_NAME_CAM)
        # Para esta prueba no necesitamos la barra de threshol
        #cv2.createTrackbar('Threshold', self.config.WINDOW_NAME_CAM, self.threshold_value, 255, self._on_threshold_change)
        
        self.menu_image = self._create_menu_image()
        cv2.imshow(self.config.WINDOW_NAME_MENU, self.menu_image)
       
    def _process_frame_and_update_buffer(self, frame):
        """
        Ahora con auto-threshold de Otsu.
        """
        # --- 1. Definición y procesamiento de la ROI grande ---
        x, y, w, h = self.big_roi_x, self.big_roi_y, self.big_roi_w, self.big_roi_h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        if not (w > 0 and h > 0): return "ROI Invalida"
        
        roi_grande = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi_grande, cv2.COLOR_BGR2GRAY)
        
        # En lugar de usar self.threshold_value, pasamos 0 y añadimos la bandera THRESH_OTSU.
        # La función ahora devuelve el threshold calculado (otsu_threshold) y la imagen.
        #thr_roi = cv2.threshold(gray_roi, self.threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
        otsu_threshold, thr_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
       
        # (Opcional) Imprimimos el valor que Otsu calculó para ver cómo se adapta.
        print(f"Otsu Threshold: {otsu_threshold}")
        # --------------------------------
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        
        # Llamamos a Tesseract para que lea el texto de nuestra ROI procesada (thr_roi)
        raw_text = pytesseract.image_to_string(thr_roi, config=custom_config)
        
        validated_text = self._validate_vaisala_reading(raw_text)
        
        if validated_text is not None:
            # Si la lectura es válida, la añadimos al búfer
            self.readings_buffer.append(validated_text)
            
            # Si el búfer excede el tamaño máximo, eliminamos el elemento más antiguo
            if len(self.readings_buffer) > self.config.READING_BUFFER_SIZE:
                self.readings_buffer.pop(0) 
        
        # Definimos un tamaño fijo para nuestras ventanas de depuración
        debug_w, debug_h = 160, 80
       
        # Redimensionamos las imágenes de la ROI (escala de grises y threshold)
        gray_debug = cv2.resize(gray_roi, (debug_w, debug_h))
        thr_debug = cv2.resize(thr_roi, (debug_w, debug_h))
       
        # Las convertimos a formato BGR (3 canales) para poder pegarlas en el frame de color
        gray_debug_bgr = cv2.cvtColor(gray_debug, cv2.COLOR_GRAY2BGR)
        thr_debug_bgr = cv2.cvtColor(thr_debug, cv2.COLOR_GRAY2BGR)
       
        # Obtenemos las dimensiones del frame principal para posicionar las vistas abajo a la derecha
        frame_h, frame_w, _ = frame.shape
       
        # Coordenadas para la vista "Escala de Grises"
        y_start1 = frame_h - debug_h - 10
        x_start1 = frame_w - (debug_w * 2) - 15
       
        # Coordenadas para la vista "Threshold"
        y_start2 = frame_h - debug_h - 10
        x_start2 = frame_w - debug_w - 10
        
        # "Pegamos" las imágenes de depuración en el frame principal
        frame[y_start1 : y_start1 + debug_h, x_start1 : x_start1 + debug_w] = gray_debug_bgr
        frame[y_start2 : y_start2 + debug_h, x_start2 : x_start2 + debug_w] = thr_debug_bgr
        
        # Añadimos texto para identificar cada vista
        cv2.putText(frame, "Grises", (x_start1, y_start1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Threshold", (x_start2, y_start2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _handle_keyboard_input(self, key):
        """Maneja las teclas para salir y para mover/redimensionar la ROI grande."""
        if key == ord('q'): return True
        if key == ord('w'): self.big_roi_y -= 5
        if key == ord('s'): self.big_roi_y += 5
        if key == ord('a'): self.big_roi_x -= 5
        if key == ord('d'): self.big_roi_x += 5
        if key == ord('e'): self.big_roi_w += 5
        if key == ord('r'): self.big_roi_h += 5
        if key == ord('c'): self.big_roi_w -= 5
        if key == ord('v'): self.big_roi_h -= 5
        if key == ord('1'):
            self.send_command("V1_ON") # Enciende la válvula 1
            print("Se Mando a abrir una electrovalvula")
        elif key == ord('2'):
            self.send_command("V1_OFF") # Apaga la válvula 1
            print("Se Mando a cerrar una electrovalvula")
        return False
    
    def _update_display(self, frame, predicted_text):
        """Actualiza el menú con la nueva predicción y refresca las ventanas."""
        updated_menu = self.menu_image.copy()
        cv2.putText(updated_menu, predicted_text, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow(self.config.WINDOW_NAME_MENU, updated_menu)
        cv2.imshow(self.config.WINDOW_NAME_CAM, frame)

    # Este metodo no lo vamos a utilizar
    #def _on_threshold_change(self, value):
    #    """Callback para la barra de threshold."""
    #    self.threshold_value = value
        
    def _create_menu_image(self):
        """Crea la imagen base para el menú de la UI."""
        menu = np.zeros((400, 500, 3), np.uint8)
        cv2.putText(menu, "Menu Principal", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(menu, "Use 'a,s,w,d' para mover la ROI", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(menu, "Use 'e,r,c,v' para el tamano", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(menu, "Presione 'q' para salir", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(menu, "Prediccion (Vaisala):", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return menu

    def _validate_vaisala_reading(self, raw_text):
        """
        Aplica las reglas de validación específicas para el display del Vaisala.
        Devuelve el texto limpio si es válido, o None si es inválido.
        """
        # Limpiamos espacios en blanco al principio y al final
        cleaned_text = raw_text.strip()

        # --- Regla 1: Debe ser un número entero y nada más ---
        # Si no es un string compuesto solo por dígitos, es inválido.
        if not cleaned_text.isdigit():
            print(f"VALIDACIÓN RECHAZADA: '{cleaned_text}' contiene caracteres no numéricos o está vacío.")
            return None

        # --- Regla 2: Debe tener 3 o 4 dígitos ---
        if not (len(cleaned_text) == 3 or len(cleaned_text) == 4):
            print(f"VALIDACIÓN RECHAZADA: '{cleaned_text}' no tiene 3 o 4 dígitos.")
            return None

        # --- Regla 3: El último dígito debe ser 0 ---
        if not cleaned_text.endswith('0'):
            print(f"VALIDACIÓN RECHAZADA: '{cleaned_text}' no termina en 0.")
            return None

        # Si el texto pasó todas las reglas, es una lectura válida.
        print(f"VALIDACIÓN ACEPTADA: '{cleaned_text}'")
        return cleaned_text
    
    def _update_stable_reading(self):
        """
        PASO 4: Analiza el búfer de lecturas y actualiza el valor estable si hay consenso.
        """
        buffer_size = len(self.readings_buffer)
        
        # No tomamos decisiones si el búfer no está razonablemente lleno
        # (ej. si tiene menos de la mitad de su capacidad máxima)
        if buffer_size < self.config.READING_BUFFER_SIZE / 2:
            return

        # Contamos cuántas veces aparece cada número en el búfer
        counts = Counter(self.readings_buffer)
        
        # Obtenemos el número más común y cuántas veces apareció
        most_common = counts.most_common(1)[0]
        candidate_value = most_common[0]
        confidence_count = most_common[1]

        # Comprobamos si la confianza (porcentaje de apariciones) supera nuestro umbral
        if (confidence_count / buffer_size) >= self.config.CONFIDENCE_THRESHOLD:
            # Si hay consenso y es un valor nuevo, lo actualizamos como estable
            if self.stable_reading != candidate_value:
                print(f"NUEVO VALOR ESTABLE: '{candidate_value}' (confianza del {confidence_count/buffer_size:.0%})")
                self.stable_reading = candidate_value
                
    def send_command(self, command):
        """
        Añade un salto de línea al final para que el ESP32 sepa cuándo termina el comando.
        """
        if self.ser and self.ser.is_open:
            # Codificamos el string a bytes y lo enviamos
            self.ser.write(f"{command}\n".encode('utf-8'))
            print(f"Comando enviado al ESP32: {command}")
        else:
            print("Error: El puerto serial no está disponible. No se envió el comando.")
        
    def cleanup(self):
        """
        Libera los recursos de la cámara y destruye las ventanas.
        Cierra la conexión serial.
        """
        print("Limpiando recursos y cerrando aplicación.")
        self.cap.release()
        cv2.destroyAllWindows()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Puerto serial cerrado.")

    def run(self):
        """Bucle principal de la aplicación."""
        print("Iniciando bucle principal...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error al capturar el frame. Saliendo.")
                break

            key = cv2.waitKey(1) & 0xFF
            if self._handle_keyboard_input(key):
                break #Presiona Q para salir
            
            if self.ser and self.ser.in_waiting > 0:
                # Leemos una línea que llega desde el ESP32
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response: # Si la respuesta no está vacía
                    print(f"Respuesta del ESP32: {response}")

            # --- NUEVO FLUJO DE PROCESAMIENTO ---
            # 1. Procesa el frame y, si encuentra un valor válido, lo guarda en el búfer.
            self._process_frame_and_update_buffer(frame)
            
            # 2. Revisa el búfer y decide si hay un nuevo valor estable.
            self._update_stable_reading()

            # 3. Actualiza el display siempre con el último valor estable.
            self._update_display(frame, self.stable_reading)

        self.cleanup()

# 4. FUNCIÓN MAIN
def main():
    """Función principal que instancia y ejecuta la aplicación."""
    app_config = Config()
    app = CalibratorApp(app_config)
    app.run()

# 5. PUNTO DE ENTRADA
if __name__ == '__main__':
    main()