# app/gui_manager.py
import tkinter as tk #Importa tkinter como tk
from tkinter import ttk #De la librería Tkinter importa ttk -> submódulo que proporciona widgets temáticos que ofrecen una apariencia más moderna y nativa en comparación con los widgets clásicos de Tkinter
from PIL import Image, ImageTk #De pilow importa Image e ImageTk
import cv2 #Importa cv2 para procesar la camara de video
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #Importa canvas para crear el grafico
from matplotlib.figure import Figure #Importa Figura de matplotlib

class GuiManager:
    def __init__(self, root, app_callbacks):
        #Ventana raiz
        self.root = root#Nombre de la ventana principal
        self.root.minsize(1200, 600)#Tamaño minimo
        self.root.geometry("1000x500-50-50")#Geometría de inicio   
        self.root.title("Sistema de Calibración Asistida")#Titulo
        
        """En Tkinter, con el método grid(), puedes configurar la posición de 
        los widgets usando row (fila) y column (columna), el espacio que un 
        widget ocupa con rowspan y columnspan, y la relación de los widgets 
        con las celdas usando sticky (para alinear y estirar) y el relleno con 
        padx y pady (externo) y ipadx y ipady (interno). Además, para el 
        redimensionamiento de la cuadrícula en sí, puedes usar los métodos 
        columnconfigure() y rowconfigure() en el widget contenedor.
        """
        self.root.columnconfigure(0, weight=1)#Cantidad de columnas
        self.root.rowconfigure(0, weight=1)#Cantidad de filas
        
        #Callbacks
        self.app_callbacks = app_callbacks
        
        self.sensor_vars = {
            'TEMP': tk.StringVar(value='--.- °C'),
            'HUM': tk.StringVar(value='--.- %'),
            'PRES': tk.StringVar(value='---- hPa'),
            'CO2': tk.StringVar(value='---- ppm'),
            'PCB1_STATE': tk.StringVar(value='UNKNOWN'),
            'PCB2_STATE': tk.StringVar(value='UNKNOWN'),
            'COOLER': tk.StringVar(value='UNKNOWN'),
            'OCR_STABLE': tk.StringVar(value='--- ppm')
        }
        self._create_widgets()

    def _create_widgets(self):
        # --- Contenedor Principal ---
        main_frame = ttk.Frame(self.root, padding="10")#Crea un frame (en la ventana raiz, separación 10p)
        main_frame.grid(row=0, column=0, sticky="nsew")#Cfg en que posición ubica el frame
        #El Administrador de Geometrías del Frame main_frame es declarado como grid
        #A continuación la configuración del grid
        # Configurar las columnas y las filas del frame principal
        main_frame.columnconfigure(0, weight=2)#Dentro de main frame, fija la col 0 peso 2
        main_frame.columnconfigure(1, weight=1)#fija col 1, mismo peso en col
        main_frame.columnconfigure(2, weight=1)#fija la 2
        # main_frame.columnconfigure(3, weight=1)#...
        main_frame.rowconfigure(0, weight=1)#Fija row 0
        main_frame.rowconfigure(1, weight=0)#Fija row 1 con peso de 0, se ajusta mas que el resto
        main_frame.rowconfigure(2, weight=1)#...

        # --- row 0, col 0: Camara ---
        camera_frame = ttk.LabelFrame(main_frame, text="Vista de la cámara", padding=5)
        camera_frame.grid(row=0, column=0,columnspan=2, sticky="nsew", padx=(0, 10), pady=(0,5))
        self.camera_label = ttk.Label(camera_frame)#Crea Label dentro de camara_frame
        self.camera_label.pack(fill="x", expand=True, anchor="center")#Administrador de geometría pack, expande
        #fill -> Define las direcciones que el widget rellene el frame al que pertenece
        #expand -> El widget se estira cuando la ventana se agranada
        #anchor -> Centra la imagen
        # --- row 1, col 0: thr + debug de la camara ---
        debug_frame = ttk.LabelFrame(main_frame, text="Debug de imagen", padding=5)
        debug_frame.grid(row=1,column=0,columnspan=2,sticky="news",padx=(0,5),pady=(5,0))
        debug_frame.columnconfigure(0, weight=1)
        debug_frame.columnconfigure(1, weight=1)
        
            # --- DENTRO DE "Debug de imagen" ---
            # --- row 0, col 0 y 1: Threshold ---
        self.thr_label = ttk.Label(debug_frame, text="Threshold:")
        # 2. Posicionar la etiqueta en una línea separada
        self.thr_label.grid(row=0, column=0, sticky="nw", padx=5, pady=5)        
        self.threshold_slider = ttk.Scale(debug_frame,
                                  from_=0, to=250,
                                  orient="horizontal",
                                  command=self.app_callbacks["on_threshold_change"])
        
        self.threshold_slider.grid(row=1,
                                   column=0,
                                   columnspan=2,
                                   padx=5,
                                   pady=2,
                                   sticky="ew")

               # --- row 1, col 0: Depuración de imagen escala de grices ---
        ttk.Label(debug_frame, text="Escala de Grises").grid(row=1, column=0, pady=(0, 2))
        self.gray_label = ttk.Label(debug_frame)
        self.gray_label.grid(row=2, column=0, padx=5,pady=5, sticky="news")
        # --- row 1, col 1: Depuración de imagen binarizada ---
        self.bin_frame = ttk.LabelFrame(debug_frame, text="Threshold (Binarizada)", padding=5)
        self.bin_frame.grid(row=2, column=1, sticky="news", padx=5, pady=5)
        self.bin_label = ttk.Label(debug_frame).grid(row=0, column=1, padx=5,pady=5, sticky="news")
        #Fin de depuración de imagen
        
        # --- row 2, col 0: Comandos OCR - Ajuste de ROI ---
        #Ajuste ROI        
        self.roi_frame = ttk.LabelFrame(main_frame, text="OCR Controls", padding=5)
        self.roi_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 10), pady=(5,0))
       # ... (código de botones ROI)
        ttk.Button(self.roi_frame, text="↑", width=3, command=lambda: self.app_callbacks["adjust_roi"]('y', -5)).grid(row=1, column=1)
        ttk.Button(self.roi_frame, text="←", width=3, command=lambda: self.app_callbacks["adjust_roi"]('x', -5)).grid(row=2, column=0)
        ttk.Button(self.roi_frame, text="→", width=3, command=lambda: self.app_callbacks["adjust_roi"]('x', 5)).grid(row=2, column=2)
        ttk.Button(self.roi_frame, text="↓", width=3, command=lambda: self.app_callbacks["adjust_roi"]('y', 5)).grid(row=3, column=1)
        ttk.Button(self.roi_frame, text="W+", width=3, command=lambda: self.app_callbacks["adjust_roi"]('w', 5)).grid(row=4, column=0, pady=(10,0))
        ttk.Button(self.roi_frame, text="W-", width=3, command=lambda: self.app_callbacks["adjust_roi"]('w', -5)).grid(row=4, column=2, pady=(10,0))
        ttk.Button(self.roi_frame, text="H+", width=3, command=lambda: self.app_callbacks["adjust_roi"]('h', 5)).grid(row=5, column=0)
        ttk.Button(self.roi_frame, text="H-", width=3, command=lambda: self.app_callbacks["adjust_roi"]('h', -5)).grid(row=5, column=2)
        
        # --- row 2, col 1: Comandos a la PC
        
        # --- Comandos del Sistema ---
        system_commands_frame = ttk.LabelFrame(main_frame, text="Comandos Sistema", padding=10)
        system_commands_frame.grid(row= 2, column=1, sticky="news", padx=5,pady=5)
        
        setpoint_frame = ttk.Frame(system_commands_frame)
        # ... (código de setpoint)
        setpoint_frame.pack(fill='x', pady=2)
        ttk.Label(setpoint_frame, text="Setpoint CO2 (ppm):").pack(side='left')
        self.setpoint_entry = ttk.Entry(setpoint_frame, width=10)
        self.setpoint_entry.pack(side='left', padx=5)
        ttk.Button(setpoint_frame, text="Enviar", command=self.app_callbacks["send_setpoint"]).pack(side='left')

        pulse_frame = ttk.Frame(system_commands_frame)
        # ... (código de pulso)
        pulse_frame.pack(fill='x', pady=2)
        ttk.Label(pulse_frame, text="Pulso Válvula (ms):").pack(side='left')
        self.pulse_entry = ttk.Entry(pulse_frame, width=10)
        self.pulse_entry.pack(side='left', padx=5)
        ttk.Button(pulse_frame, text="Enviar", command=self.app_callbacks["send_pulse"]).pack(side='left')
        
        
        buttons_frame = ttk.Frame(system_commands_frame)
        # ... (código de botones de acción)
        buttons_frame.pack(fill='x', pady=(10, 2))
        ttk.Button(buttons_frame, text="Apagar/Prender Cooler", command=lambda: self.app_callbacks["send_command"]("TOGGLE_COOLER")).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(buttons_frame, text="Calibrar Sensor", command=lambda: self.app_callbacks["send_command"]("CALIBRATE_SENSOR")).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(buttons_frame, text="PÁNICO", command=lambda: self.app_callbacks["send_command"]("OPEN_ALL")).pack(side='left', expand=True, fill='x', padx=2)

        
        # --- row 0 y 1, col 1: Grafico ---
        
        # --- row 2, col 1
        
        
        # --- Columna 2: Dashboard ---
        dashboard_frame = ttk.LabelFrame(main_frame, text="Dashboard", padding=10)
        dashboard_frame.grid(row=0, column=1, sticky="ew", padx=5, pady=(0,5))
        
        # --- Llenar Dashboard --- -> Crear funcion
        ttk.Label(dashboard_frame, text="Temperatura:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(dashboard_frame, textvariable=self.sensor_vars['TEMP']).grid(row=0, column=1, sticky="w", pady=2, padx=5)
        ttk.Label(dashboard_frame, text="Humedad:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(dashboard_frame, textvariable=self.sensor_vars['HUM']).grid(row=1, column=1, sticky="w", pady=2, padx=5)
        ttk.Label(dashboard_frame, text="Presión:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(dashboard_frame, textvariable=self.sensor_vars['PRES']).grid(row=2, column=1, sticky="w", pady=2, padx=5)
        ttk.Label(dashboard_frame, text="CO2 (Sensor):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(dashboard_frame, textvariable=self.sensor_vars['CO2']).grid(row=3, column=1, sticky="w", pady=2, padx=5)
        ttk.Label(dashboard_frame, text="CO2 (OCR):").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(dashboard_frame, textvariable=self.sensor_vars['OCR_STABLE'], font=("Helvetica", 12, "bold")).grid(row=4, column=1, sticky="w", pady=2, padx=5)
        ttk.Separator(dashboard_frame, orient='horizontal').grid(row=5, columnspan=2, sticky='ew', pady=10)
        ttk.Label(dashboard_frame, text="Estado PCB1:").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Label(dashboard_frame, textvariable=self.sensor_vars['PCB1_STATE']).grid(row=6, column=1, sticky="w", pady=2, padx=5)
        ttk.Label(dashboard_frame, text="Estado Cooler:").grid(row=7, column=0, sticky="w", pady=2)
        ttk.Label(dashboard_frame, textvariable=self.sensor_vars['COOLER']).grid(row=7, column=1, sticky="w", pady=2, padx=5)
        ttk.Label(dashboard_frame, text="Estado PCB2:").grid(row=8, column=0, sticky="w", pady=2)
        ttk.Label(dashboard_frame, textvariable=self.sensor_vars['PCB2_STATE']).grid(row=8, column=1, sticky="w", pady=2, padx=5)
        
        # --- Fila 2 (fusionada): Gráfico en Tiempo Real ---
        plot_frame = ttk.LabelFrame(main_frame, text="Gráfico en tiempo real de las variables", padding=5)
        plot_frame.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=5, pady=(5,0))
        # (El código para crear el gráfico no cambia)
        # --- Llenar Gráfico ---
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.line_sensor, = self.ax.plot([], [], 'r-', label='Sensor', linewidth=1.5)
        self.line_ocr, = self.ax.plot([], [], 'b--', label='Patrón OCR', linewidth=1.5)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    #Funcion _create_widgets
    
    #Aca debería poner las funciones del threshold, los botones, los comandos, el grafico, etc, lo de arriba solo arma la pantalla    
    
    def update_plot(self, sensor_data, ocr_data):
        self.line_sensor.set_data(range(len(sensor_data)), list(sensor_data))
        self.line_ocr.set_data(range(len(ocr_data)), list(ocr_data))
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.canvas.draw()
        
    def update_camera_feed(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.image = imgtk
        self.camera_label.configure(image=imgtk)

    def update_debug_images(self, gray_roi, thresh_roi):
        if gray_roi is not None:
            img_gray = Image.fromarray(gray_roi)
            imgtk_gray = ImageTk.PhotoImage(image=img_gray.resize((160, 80)))
            self.gray_label.image = imgtk_gray
            self.gray_label.configure(image=imgtk_gray)
        
        if thresh_roi is not None:
            img_thresh = Image.fromarray(thresh_roi)
            imgtk_thresh = ImageTk.PhotoImage(image=img_thresh.resize((160, 80)))
            self.thr_label.image = imgtk_thresh
            self.thr_label.configure(image=imgtk_thresh)
            
    def update_sensor_data(self, sensor_data, stable_reading):
        self.sensor_vars['TEMP'].set(f"{sensor_data.get('TEMP', '--.-')} °C")
        self.sensor_vars['HUM'].set(f"{sensor_data.get('HUM', '--.-')} %")
        self.sensor_vars['PRES'].set(f"{sensor_data.get('PRES', '----')} hPa")
        self.sensor_vars['CO2'].set(f"{sensor_data.get('CO2', '----')} ppm")
        self.sensor_vars['PCB1_STATE'].set(sensor_data.get('PCB1_STATE', 'UNKNOWN'))
        self.sensor_vars['PCB2_STATE'].set(sensor_data.get('PCB2_STATE', 'UNKNOWN'))
        self.sensor_vars['COOLER'].set(sensor_data.get('COOLER', 'UNKNOWN'))
        self.sensor_vars['OCR_STABLE'].set(f"{stable_reading} ppm")