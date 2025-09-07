# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 00:26:26 2025

@author: paco2
"""

# app/calibrator_app.py
import cv2
import logging
from datetime import datetime
from .serial_manager import SerialManager
from .ocr_manager import OCRManager
from .ui_manager import UIManager

class CalibratorApp:
    def __init__(self, config, data_logger):
        self.config = config
        self.data_logger = data_logger

        # Inicializar manejadores
        self.serial_manager = SerialManager(config['serial']['port'], config['serial']['baud_rate'])
        self.ocr_manager = OCRManager(config)
        self.ui_manager = UIManager(config['window_names'])

        # Estado de la aplicación
        self.sensor_data = {
            'TEMP': '--.-', 'HUM': '--.-', 'PRES': '----', 'CO2': '----',
            'PCB2_STATE': 'UNKNOWN', 'PCB1_STATE': 'UNKNOWN', 'COOLER': 'UNKNOWN'
        }
        roi_cfg = config['detection']['roi']
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi_cfg.values()
        self.threshold = 150

    def setup(self):
        """Configura la cámara y las ventanas de UI."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.critical("No se puede abrir la cámara. La aplicación no puede continuar.")
            raise IOError("No se puede abrir la cámara")
        
        self.ui_manager.setup_windows(self.threshold, lambda v: setattr(self, 'threshold', v))

    def run(self):
        """Bucle principal de la aplicación."""
        logging.info("Iniciando bucle principal...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("No se pudo leer el frame. Reintentando...")
                continue

            # Gestionar entrada de teclado
            key = cv2.waitKey(1) & 0xFF
            if self._handle_keyboard_input(key):
                break

            # Procesos principales
            self.serial_manager.connect()
            self._process_serial_data()
            self._process_ocr(frame)
            
            # Dibujar ROI y actualizar pantalla
            cv2.rectangle(frame, (self.roi_x, self.roi_y), (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (255, 0, 0), 1)
            self.ui_manager.update_display(frame, self.sensor_data, self.ocr_manager.stable_reading, self.debug_images)

        self.cleanup()

    def _process_serial_data(self):
        """Lee y procesa los datos del puerto serie."""
        line = self.serial_manager.read_line()
        if not line: return
        
        logging.debug(f"Datos recibidos: {line}")
        try:
            for pair in line.split(';'):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    if key in self.sensor_data:
                        self.sensor_data[key] = value
            # Loguear datos si el paquete está completo
            if all(k in line for k in ['STATUS', 'TEMP', 'CO2']):
                self._log_sensor_data()
        except Exception as e:
            logging.warning(f"No se pudo parsear la trama '{line}': {e}")
            
    def _log_sensor_data(self):
        now = datetime.now()
        log_line = (
            f"{now.strftime('%d/%m/%Y')},{now.strftime('%H:%M:%S')},"
            f"Medicion_ESP32,{self.sensor_data.get('CO2', '')},ppm,"
            f"{self.sensor_data.get('TEMP', '')},{self.sensor_data.get('HUM', '')},"
            f"{self.sensor_data.get('PRES', '')}"
        )
        self.data_logger.info(log_line)

    def _process_ocr(self, frame):
        """Procesa el frame para OCR y actualiza el valor estable."""
        roi_coords = (self.roi_x, self.roi_y, self.roi_w, self.roi_h)
        self.debug_images = self.ocr_manager.process_frame(frame, roi_coords, self.threshold)
        
        if self.ocr_manager.update_stable_reading():
            now = datetime.now()
            log_line = (
                f"{now.strftime('%d/%m/%Y')},{now.strftime('%H:%M:%S')},"
                f"Medicion_VAISALA,{self.ocr_manager.stable_reading},ppm,,,"
            )
            self.data_logger.info(log_line)
            
    def _handle_keyboard_input(self, key):
        """Maneja las teclas para controlar la aplicación."""
        if key == ord('q'): return True
        # Controles de ROI
        if key == ord('w'): self.roi_y -= 5
        if key == ord('s'): self.roi_y += 5
        if key == ord('a'): self.roi_x -= 5
        if key == ord('d'): self.roi_x += 5
        # ... (otros controles de ROI)
        
        # Comandos para el ESP32
        if key == ord('z'): self.serial_manager.send_command("SET_CO2(800)")
        if key == ord('k'): self.serial_manager.send_command("CALIBRATE_SENSOR")
        if key == ord('p'): self.serial_manager.send_command("OPEN_ALL")
        if key == ord('t'): self.serial_manager.send_command("TOGGLE_COOLER")
        
        return False
        
    def cleanup(self):
        """Libera recursos al cerrar."""
        logging.info("Limpiando recursos y cerrando.")
        self.cap.release()
        self.ui_manager.close_all_windows()
        self.serial_manager.close()