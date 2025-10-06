# app/utils.py
import logging
import os
import yaml
from datetime import datetime

PCB2_STATE_MAP = {
    0: "IDLE",
    1: "EXECUTING_SETPOINT",
    2: "SETPOINT_STABLE",
    3: "EXECUTING_CALIBRATION",
    4: "PULSE",
    5: "PANIC_MODE"
}

def setup_loggers():
    """Configura el logger de la aplicación y el logger de datos CSV."""
    # (El código de la función setup_loggers() va aquí, sin cambios)
    # --- Configuración del Logger Principal (calibrator.log) ---
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        filename='calibrator.log',
                        filemode='a')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)

    # --- Configuración del Logger de Datos (mediciones.csv) ---
    data_filename = 'data_logger.csv'
    file_exists = os.path.exists(data_filename)
    data_logger = logging.getLogger('data_logger')
    data_logger.setLevel(logging.INFO)
    
    if data_logger.hasHandlers():
        data_logger.handlers.clear()
        
    data_handler = logging.FileHandler(data_filename, mode='a', encoding='utf-8')
    data_handler.setFormatter(logging.Formatter('%(message)s'))
    data_logger.addHandler(data_handler)
    
    if not file_exists:
        data_logger.info("Fecha,Hora,GM-70,MH-Z19C,Temperatura,Humedad,Presion")

    return data_logger

def load_config(config_path='config.yaml'):
    """Lee y carga la configuración desde un archivo YAML."""
    # (El código de la función load_config() va aquí, sin cambios)
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo de configuración: {config_path}")
        return None
    except Exception as e:
        logging.error(f"Error al leer o parsear el archivo de configuración: {e}")
        return None
    
