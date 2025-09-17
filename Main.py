# main.py
import logging
import tkinter as tk
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
        root = tk.Tk()
        app = CalibratorApp(config, data_logger, root)
        
        root.protocol("WM_DELETE_WINDOW", app.cleanup) # Al hacer clic en la 'X'
        root.bind('<q>', lambda event: app.cleanup())   # Al presionar 'q'
     
        if app.setup():
            app.run()
            
    except Exception as e:
        logging.critical(f"Ha ocurrido un error fatal: {e}", exc_info=True)

if __name__ == '__main__':
    main()