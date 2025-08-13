# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 21:00:00 2025
Versión del calibrador automático utilizando Tesseract OCR con lógica de decisión estable.
@author: paco2
"""

# 1. IMPORTS
import cv2
import numpy as np
import pytesseract
import serial
import time
from collections import Counter
import yaml


# 2. CONFIGURACIÓN
# 2. FUNCIÓN PARA CARGAR CONFIGURACIÓN (Reemplaza la clase Config)
def load_config(config_path='config.yaml'):
    """
    Lee y carga la configuración desde un archivo YAML.
    Devuelve un diccionario con la configuración o None si hay un error.
    """
    try:
        # Usamos 'with' para asegurarnos de que el archivo se cierre correctamente
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("--- ERROR CRÍTICO ---")
        print(f"No se encontró el archivo de configuración: {config_path}")
        print("Asegurate de que 'config.yaml' exista en la misma carpeta que el script.")
        print("--------------------")
        return None
    except Exception as e:
        print("--- ERROR CRÍTICO ---")
        print(f"Error al leer o parsear el archivo de configuración: {e}")
        print("--------------------")
        return None

# 3. CLASE PRINCIPAL DE LA APLICACIÓN
class CalibratorApp:
    """Clase principal que encapsula toda la lógica de la aplicación de calibración."""
    def __init__(self, config_data):
        """Constructor: Inicializa todos los componentes de la aplicación."""
        self.config = config_data
        
        # --- Variables de Estado ---
        self.readings_buffer = []
        self.stable_reading = "---"
        self.sensor_data = {'TEMP': '--.-', 'HUM': '--.-', 'PRES': '----'}
        self.ser = None
        self.last_reconnect_attempt = 0
        self.cap = None
        self.menu_image_base = None

        # --- Inicialización de Componentes ---
        self._initialize_tesseract()
        self._initialize_camera()
        self._initialize_roi_state()
        self._setup_ui()
    
    def _initialize_tesseract(self):
        """Configura la ruta de Tesseract desde la config."""
        try:
            # --- CAMBIO IMPORTANTE ---
            # Accedemos a la configuración como un diccionario anidado
            tesseract_path = self.config['tesseract']['command_path']
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print(f"--- ERROR DE TESSERACT --- \nNo se pudo encontrar Tesseract. Revisá la ruta en config.yaml\nError: {e}\n--------------------------")

    def _initialize_camera(self):
        """Inicializa la captura de video."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("No se puede abrir la cámara")
            
    def _initialize_roi_state(self):
        """Inicializa las variables de la ROI desde la config."""
        # Accedemos a los valores del diccionario de configuración
        roi_config = self.config['detection']['roi']
        self.big_roi_x = roi_config['initial_x']
        self.big_roi_y = roi_config['initial_y']
        self.big_roi_w = roi_config['initial_width']
        self.big_roi_h = roi_config['initial_height']
        self.threshold_value = 150

    
    def _setup_ui(self):
        """Configura las ventanas y los controles de OpenCV."""
        
        cam_window_name = self.config['window_names']['camera']
        menu_window_name = self.config['window_names']['menu']
        cv2.namedWindow(cam_window_name)
        cv2.createTrackbar('Threshold', cam_window_name, self.threshold_value, 255, self._on_threshold_change)
        self.menu_image_base = self._create_menu_image()
        cv2.imshow(menu_window_name, self.menu_image_base)
        
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
                break
            
            self._manage_serial_connection()
            self._read_and_parse_serial()
            
            brightness_ratio = self._process_frame_and_update_buffer(frame)
            self._update_stable_reading()
            self._update_display(frame, self.stable_reading, brightness_ratio)

        self.cleanup()

    def _process_frame_and_update_buffer(self, frame):
        """Procesa el frame, actualiza el búfer y devuelve el ratio de brillo."""
        x, y, w, h = self.big_roi_x, self.big_roi_y, self.big_roi_w, self.big_roi_h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        brightness_ratio = 0.0
        if w > 0 and h > 0:
            roi_grande = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(roi_grande, cv2.COLOR_BGR2GRAY)
            thr_roi = cv2.threshold(gray_roi, self.threshold_value, 255, cv2.THRESH_BINARY_INV)[1]

            if thr_roi.size > 0:
                white_pixels = cv2.countNonZero(thr_roi)
                brightness_ratio = white_pixels / thr_roi.size
            
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            raw_text = pytesseract.image_to_string(thr_roi, config=custom_config)
            validated_text = self._validate_vaisala_reading(raw_text)
            
            if validated_text is not None:
                self.readings_buffer.append(validated_text)
                if len(self.readings_buffer) > self.config['validation_buffer']['size']:
                    self.readings_buffer.pop(0)

            # --- Visualización de Depuración ---
            debug_w, debug_h = 160, 80
            gray_debug = cv2.resize(gray_roi, (debug_w, debug_h))
            thr_debug = cv2.resize(thr_roi, (debug_w, debug_h))
            gray_debug_bgr = cv2.cvtColor(gray_debug, cv2.COLOR_GRAY2BGR)
            thr_debug_bgr = cv2.cvtColor(thr_debug, cv2.COLOR_GRAY2BGR)
            
            frame_h, frame_w, _ = frame.shape
            y_start = frame_h - debug_h - 10
            x_start1 = frame_w - (debug_w * 2) - 15
            x_start2 = frame_w - debug_w - 10
            
            frame[y_start : y_start + debug_h, x_start1 : x_start1 + debug_w] = gray_debug_bgr
            frame[y_start : y_start + debug_h, x_start2 : x_start2 + debug_w] = thr_debug_bgr
            
            cv2.putText(frame, "Grises", (x_start1, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Threshold", (x_start2, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return brightness_ratio

    def _update_stable_reading(self):
        """Analiza el búfer de lecturas y actualiza el valor estable si hay consenso."""
        buffer_size = len(self.readings_buffer)
        if buffer_size < self.config['detection']['validation_buffer']['size'] / 2:
            return

        counts = Counter(self.readings_buffer)
        if not counts: return
        
        most_common = counts.most_common(1)[0]
        candidate_value, confidence_count = most_common

        if (confidence_count / buffer_size) >= self.config['detection']['validation_buffer']['confidence_threshold']:
            if self.stable_reading != candidate_value:
                print(f"NUEVO VALOR ESTABLE: '{candidate_value}' (confianza del {confidence_count/buffer_size:.0%})")
                self.stable_reading = candidate_value

    def _update_display(self, frame, stable_reading, brightness_ratio):
        """Actualiza el menú con todos los datos dinámicos."""
        updated_menu = self.menu_image_base.copy()
        
        cv2.putText(updated_menu, stable_reading, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        cv2.putText(updated_menu, "Temperatura:", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(updated_menu, f"{self.sensor_data['TEMP']} C", (250, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(updated_menu, "Humedad:", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(updated_menu, f"{self.sensor_data['HUM']} %", (250, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(updated_menu, "Presion:", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(updated_menu, f"{self.sensor_data['PRES']} hPa", (250, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(updated_menu, "Ratio Px Blancos:", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(updated_menu, f"{brightness_ratio:.2%}", (250, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(self.config['window_names']['menu'], updated_menu)
        cv2.imshow(self.config['window_names']['camera'], frame)

    # --- Métodos de Ayuda y Lógica Interna ---
    
    def _validate_vaisala_reading(self, raw_text):
        """Aplica las reglas de validación específicas para el display del Vaisala."""
        cleaned_text = raw_text.strip()
        if not cleaned_text.isdigit(): return None
        if not (len(cleaned_text) == 3 or len(cleaned_text) == 4): return None
        if not cleaned_text.endswith('0'): return None
        return cleaned_text

    def _manage_serial_connection(self):
        """Verifica el puerto serial y reconecta si es necesario, usando la config."""
        if self.ser and self.ser.is_open:
            return

        current_time = time.time()
        if current_time - self.last_reconnect_attempt > 10:
            # Leemos los parámetros del diccionario de configuración
            port = self.config['serial']['port']
            baud = self.config['serial']['baud_rate']
            
            print(f"Intentando conectar al puerto serial {port}...")
            self.last_reconnect_attempt = current_time
            try:
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)
                print(f"¡Puerto serial {port} conectado exitosamente!")
            except serial.SerialException:
                print("Conexión fallida. Se reintentará...")
                self.ser = None

    def _read_and_parse_serial(self):
        """Lee una línea del puerto serial de forma segura y actualiza los datos."""
        if not (self.ser and self.ser.is_open): return
        try:
            if self.ser.in_waiting > 0:
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    pairs = response.split(';')
                    for pair in pairs:
                        if ':' in pair:
                            key, value = pair.split(':')
                            if key in self.sensor_data:
                                self.sensor_data[key] = value
        except serial.SerialException:
            self._handle_disconnect()
        except Exception as e:
            print(f"Error inesperado en lectura serial: {e}")
            self._handle_disconnect()

    def send_command(self, command):
        """Envía un comando al ESP32 de forma segura."""
        if not (self.ser and self.ser.is_open):
            print("Error: El puerto serial no está disponible. No se envió el comando.")
            return
        try:
            self.ser.write(f"{command}\n".encode('utf-8'))
            print(f"Comando enviado al ESP32: {command}")
        except serial.SerialException:
            self._handle_disconnect()

    def _handle_disconnect(self):
        """Centraliza la lógica para manejar una desconexión del ESP32."""
        if self.ser and self.ser.is_open:
            print("--- CONEXIÓN SERIAL PERDIDA ---")
            self.ser.close()
        self.ser = None
        self.last_reconnect_attempt = time.time()

    def _handle_keyboard_input(self, key):
        """Maneja las teclas para salir y para controlar la aplicación."""
        if key == ord('q'): return True
        if key == ord('w'): self.big_roi_y -= 5
        if key == ord('s'): self.big_roi_y += 5
        if key == ord('a'): self.big_roi_x -= 5
        if key == ord('d'): self.big_roi_x += 5
        if key == ord('e'): self.big_roi_w += 5
        if key == ord('r'): self.big_roi_h += 5
        if key == ord('c'): self.big_roi_w -= 5
        if key == ord('v'): self.big_roi_h -= 5
        if key == ord('1'): self.send_command("V1_ON")
        elif key == ord('2'): self.send_command("V1_OFF")
        return False

    def _on_threshold_change(self, value):
        """Callback para la barra de threshold."""
        self.threshold_value = value
        
    def _create_menu_image(self):
        """Crea la imagen base para el NUEVO menú."""
        menu = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(menu, "Sistema de Calibracion Asistida", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.line(menu, (30, 70), (610, 70), (255, 255, 255), 1)
        cv2.putText(menu, "Lectura OCR", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        cv2.putText(menu, "Datos de Sensores (ESP32)", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        cv2.putText(menu, "Depuracion", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        return menu

    def cleanup(self):
        """Libera los recursos y cierra la conexión serial."""
        print("Limpiando recursos y cerrando aplicación.")
        self.cap.release()
        cv2.destroyAllWindows()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Puerto serial cerrado.")

# 4. FUNCIÓN MAIN
def main():
    """Función principal que instancia y ejecuta la aplicación."""
    print("Cargando configuración desde 'config.yaml'...")
    # Llamamos a nuestra nueva función para leer el archivo
    config_data = load_config()
    
    # Si la carga de configuración falla (ej. archivo no encontrado), no continuamos.
    if config_data is None:
        input("Presioná Enter para salir.") # Pausa para que el usuario pueda leer el error
        return
    
    app = CalibratorApp(config_data)
    app.run()

# 5. PUNTO DE ENTRADA
if __name__ == '__main__':
    main()