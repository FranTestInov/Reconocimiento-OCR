# main.py
import logging
from app.calibrator_app import CalibratorApp
from app.utils import load_config, setup_loggers

def main():
    """Punto de entrada principal de la aplicación."""
    data_logger = setup_loggers()
    
    logging.info("Cargando configuración desde 'config.yaml'...")
    config = load_config()
    if not config:
        logging.critical("La carga de configuración falló. La aplicación no puede continuar.")
        return
    
    try:
        app = CalibratorApp(config, data_logger)
        app.setup()
        app.run()
    except Exception as e:
        logging.critical(f"Ha ocurrido un error fatal: {e}", exc_info=True)

if __name__ == '__main__':
    main()