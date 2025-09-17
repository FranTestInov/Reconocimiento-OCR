# app/calibrator_app.py
import cv2
import logging
from datetime import datetime
from tkinter import messagebox
from collections import deque
from .serial_manager import SerialManager
from .ocr_manager import OCRManager
from .gui_manager import GuiManager

class CalibratorApp:
    def __init__(self, config, data_logger, root):
        self.config = config
        self.data_logger = data_logger
        self.root = root

        self.serial_manager = SerialManager(config['serial']['port'], config['serial']['baud_rate'])
        self.ocr_manager = OCRManager(config)
        
        callbacks = {
            "on_threshold_change": self.on_threshold_change,
            "adjust_roi": self.adjust_roi,
            "send_command": self.serial_manager.send_command,
            "send_setpoint": self.send_setpoint_command,
            "send_pulse": self.send_pulse_command
        }
        self.gui_manager = GuiManager(self.root, callbacks)
        
        self.sensor_data = {
            'TEMP': '--.-', 'HUM': '--.-', 'PRES': '----', 'CO2': '----',
            'PCB2_STATE': 'UNKNOWN', 'PCB1_STATE': 'UNKNOWN', 'COOLER': 'UNKNOWN'
        }
        roi_cfg = config['detection']['roi']
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi_cfg.values()
        self.threshold = 150
        self.debug_images = None
        
        # === INICIO DE LA CORRECCIÓN 1: Asegurarse de que el historial de datos se inicialice ===
        # Usamos deque para tener listas de tamaño fijo.
        self.plot_data_sensor = deque(maxlen=50) # Guardar las últimas 50 muestras
        self.plot_data_ocr = deque(maxlen=50)
        # === FIN DE LA CORRECCIÓN 1 ===

    def setup(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.critical("No se puede abrir la cámara.")
            return False
        self.gui_manager.threshold_slider.set(self.threshold)
        return True

    def run(self):
        self.update_loop() 
        self.root.mainloop()

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(20, self.update_loop)
            return

        self.serial_manager.connect()
        self._process_serial_data()
        self._process_ocr(frame)
        
        cv2.rectangle(frame, (self.roi_x, self.roi_y), (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (255, 0, 0), 1)
        
        self.gui_manager.update_camera_feed(frame)
        self.gui_manager.update_sensor_data(self.sensor_data, self.ocr_manager.stable_reading)
        
        if self.debug_images:
            self.gui_manager.update_debug_images(self.debug_images[0], self.debug_images[1])
        
        # === INICIO DE LA CORRECCIÓN 3: Llamar a la función para actualizar el gráfico ===
        self.gui_manager.update_plot(self.plot_data_sensor, self.plot_data_ocr)
        # === FIN DE LA CORRECCIÓN 3 ===

        self.root.after(20, self.update_loop)
        

    # --- MÉTODOS CALLBACK para la GUI ---
    def on_threshold_change(self, value):
        """Se ejecuta cuando el slider de threshold cambia."""
        self.threshold = int(float(value))

    def adjust_roi(self, part, delta):
        """Ajusta la posición o tamaño de la ROI."""
        if part == 'x': self.roi_x += delta
        elif part == 'y': self.roi_y += delta
        elif part == 'w': self.roi_w += delta
        elif part == 'h': self.roi_h += delta
        self.roi_w = max(10, self.roi_w) # Evitar tamaño negativo
        self.roi_h = max(10, self.roi_h) # Evitar tamaño negativo

    def send_setpoint_command(self):
        """Lee el valor del Entry de setpoint, lo valida y envía el comando."""
        value_str = self.gui_manager.setpoint_entry.get()
        if not value_str.isdigit():
            messagebox.showerror("Error de Entrada", "El valor del setpoint debe ser un número entero.")
            return
        self.serial_manager.send_command(f"SET_CO2({value_str})")

    def send_pulse_command(self):
        """Lee el valor del Entry de pulso, lo valida y envía el comando."""
        value_str = self.gui_manager.pulse_entry.get()
        if not value_str.isdigit():
            messagebox.showerror("Error de Entrada", "El valor del pulso debe ser un número entero (en ms).")
            return
        self.serial_manager.send_command(f"PULSE({value_str})")
        
    def _process_ocr(self, frame):
        roi_coords = (self.roi_x, self.roi_y, self.roi_w, self.roi_h)
        self.debug_images = self.ocr_manager.process_frame(frame, roi_coords, self.threshold)
        
        if self.ocr_manager.update_stable_reading():
            self._log_ocr_data()
            
            # === INICIO DE LA CORRECCIÓN 2: Añadir datos del OCR al historial del gráfico ===
            stable_value_str = self.ocr_manager.stable_reading
            if stable_value_str.isdigit():
                self.plot_data_ocr.append(int(stable_value_str))
            # === FIN DE LA CORRECCIÓN 2 ===

    def _process_serial_data(self):
        line = self.serial_manager.read_line()
        if not line: return
        
        try:
            temp_data = {}
            for pair in line.split(';'):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    if key in self.sensor_data:
                        temp_data[key] = value
            
            self.sensor_data.update(temp_data)

            # === INICIO DE LA CORRECCIÓN 2: Añadir datos del sensor al historial del gráfico ===
            if 'CO2' in temp_data and temp_data['CO2'].isdigit():
                self.plot_data_sensor.append(int(temp_data['CO2']))
            # === FIN DE LA CORRECCIÓN 2 ===
            
            if all(k in line for k in ['PCB2_STATE', 'TEMP', 'CO2']):
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

    def _log_ocr_data(self):
        now = datetime.now()
        log_line = (
            f"{now.strftime('%d/%m/%Y')},{now.strftime('%H:%M:%S')},"
            f"Medicion_VAISALA,{self.ocr_manager.stable_reading},ppm,,,"
        )
        self.data_logger.info(log_line)
            
    def cleanup(self):
        """Libera recursos al cerrar la ventana."""
        logging.info("Limpiando recursos y cerrando.")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.serial_manager.close()
        self.root.destroy()
