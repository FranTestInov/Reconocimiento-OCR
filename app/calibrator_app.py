# app/calibrator_app.py
import cv2
import logging
from datetime import datetime
from tkinter import messagebox
from collections import deque
from .serial_manager import SerialManager
from .ocr_manager import OCRManager
from .gui_manager import GuiManager
from .utils import PCB2_STATE_MAP

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
            "send_pulse": self.send_pulse_command,
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
        
        # Usamos deque para tener listas de tamaño fijo.
        self.plot_data_sensor = deque(maxlen=1000) # Guardar las últimas 50 muestras
        self.plot_data_ocr = deque(maxlen=1000)
        
        self.last_ocr_value = 400

    def setup(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.critical("No se puede abrir la cámara.")
            return False
        self.gui_manager.threshold_slider.set(self.threshold)
        #logging.INFO("Camapra abierta.")
        return True

    def run(self):
        self.update_loop()
        self._update_plot_periodically()
        self.root.mainloop()

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(20, self.update_loop)
            return

        self.serial_manager.connect()
        self._process_serial_data()
        ancho_deseado = 640
        alto_deseado = 480
        frame = cv2.resize(frame, (ancho_deseado, alto_deseado))
        self._process_ocr(frame)
        
        cv2.rectangle(frame, (self.roi_x, self.roi_y), (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (255, 0, 0), 1)
        
        ancho_deseado = 400
        alto_deseado = 300
        frame_redimensionado = cv2.resize(frame, (ancho_deseado, alto_deseado))
        
        self.gui_manager.update_camera_feed(frame_redimensionado)
        self.gui_manager.update_sensor_data(self.sensor_data, self.ocr_manager.stable_reading)
        
        if self.debug_images:
            self.gui_manager.update_debug_images(self.debug_images[0], self.debug_images[1])
        
        #self.gui_manager.update_plot(self.plot_data_sensor, self.plot_data_ocr)

        self.root.after(20, self.update_loop)
        

    def _update_plot_periodically(self):
        """Actualiza el gráfico cada segundo."""
        self.gui_manager.update_plot(self.plot_data_sensor, self.plot_data_ocr)
        self._log_sensor_data() # Loguea las mediciones en el CSV cada 1 segundo
        self.root.after(1000, self._update_plot_periodically) # Llama a este mismo método después de 1000ms
        
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
            #self._log_ocr_data()
            
            stable_value_str = self.ocr_manager.stable_reading
            if stable_value_str.isdigit():
                    self.last_ocr_value = int(stable_value_str)
                
    def _process_serial_data(self):
        
            line = self.serial_manager.read_line()
            if not line: return
    
            # Diferenciamos entre tramas de datos (telemetría) y mensajes de log (eventos)
            # Asumimos que la trama de telemetría SIEMPRE comienza con "PCB2_STATE:"
            if line.startswith("PCB2_STATE:"):
                try:
                    # --- Es una Trama de Telemetría ---
                    temp_data = {}
                    for pair in line.split(';'):
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            if key in self.sensor_data:
                                if key == 'PCB2_STATE':
                                    try:
                                        state_num = int(value)
                                        # Usamos .get() para obtener el nombre.
                                        # Si no existe, devuelve 'UNKNOWN'.
                                        temp_data[key] = PCB2_STATE_MAP.get(state_num, 'UNKNOWN')
                                    except (ValueError, TypeError):
                                        temp_data[key] = 'INVALID_STATE' # Si el valor no es un número
                                else:
                                    temp_data[key] = value
                                    
                    self.sensor_data.update(temp_data)
    
                    # Actualizamos los datos para el gráfico
                    if 'CO2' in temp_data and temp_data['CO2'].isdigit():
                        self.plot_data_sensor.append(int(temp_data['CO2']))
                        self.plot_data_ocr.append(self.last_ocr_value)
                    
                except Exception as e:
                    # Si falla el parseo de una trama que *parecía* telemetría, es un error
                    logging.error(f"Error al parsear la trama de telemetría '{line}': {e}")
            
            elif line:
                # --- Es un Mensaje de Evento/Log ---
                # Si no es telemetría, es un mensaje de log/evento del ESP32
                # Lo registramos en el log de eventos principal (calibrator.log)
                # Usamos el logger raíz configurado en utils.py
                logging.info(f"[ESP32-Cliente]: {line}")

            
    def _log_sensor_data(self):
        now = datetime.now()
        log_line = (
            f"{now.strftime('%d/%m/%Y')},{now.strftime('%H:%M:%S')}," #TimeStamp
            f"{self.last_ocr_value}," #CO2 Vaisala
            f"{self.sensor_data.get('CO2', '')}," #CO2 MH-Z19C
            f"{self.sensor_data.get('TEMP', '')}," #Temp
            f"{self.sensor_data.get('HUM', '')}," #Hum
            f"{self.sensor_data.get('PRES', '')}" #Presion
        )
        self.data_logger.info(log_line)

    def _log_ocr_data(self):
        now = datetime.now()
        log_line = (
            f"{now.strftime('%d/%m/%Y')},{now.strftime('%H:%M:%S')},"
            f"Medicion_VAISALA,{self.ocr_manager.stable_reading},ppm"
        )
        self.data_logger.info(log_line)
            
    def cleanup(self):
        """Libera recursos al cerrar la ventana."""
        logging.info("Limpiando recursos y cerrando.")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.serial_manager.close()
        self.root.destroy()
