# app/serial_manager.py
import serial
import time
import logging

class SerialManager:
    def __init__(self, port, baud_rate):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.last_reconnect_attempt = 0

    def connect(self):
        """Intenta conectar o reconectar al puerto serial."""
        if self.ser and self.ser.is_open:
            return True

        current_time = time.time()
        if current_time - self.last_reconnect_attempt > 10:
            logging.info(f"Intentando conectar al puerto serial {self.port}...")
            self.last_reconnect_attempt = current_time
            try:
                self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
                time.sleep(2)
                logging.info(f"¡Puerto serial {self.port} conectado exitosamente!")
                return True
            except serial.SerialException:
                logging.warning(f"Conexión a {self.port} fallida. Se reintentará...")
                self.ser = None
        return False

    def read_line(self):
        """Lee una línea del puerto serial si hay datos disponibles."""
        if not (self.ser and self.ser.is_open and self.ser.in_waiting > 0):
            return None
        try:
            return self.ser.readline().decode('utf-8', errors='ignore').strip()
        except serial.SerialException:
            self._handle_disconnect()
            return None

    def send_command(self, command):
        """Envía un comando al ESP32."""
        if not (self.ser and self.ser.is_open):
            logging.warning(f"Envío de '{command}' fallido: Puerto no disponible.")
            return
        try:
            self.ser.write(f"{command}\n".encode('utf-8'))
            logging.info(f"Comando enviado al ESP32: {command}")
        except serial.SerialException:
            self._handle_disconnect()

    def _handle_disconnect(self):
        """Maneja una desconexión inesperada."""
        if self.ser and self.ser.is_open:
            logging.warning("CONEXIÓN SERIAL PERDIDA")
            self.ser.close()
        self.ser = None
        self.last_reconnect_attempt = time.time()

    def close(self):
        """Cierra la conexión serial de forma segura."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            logging.info("Puerto serial cerrado.")