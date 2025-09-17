# app/gui_manager.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class GuiManager:
    def __init__(self, root, app_callbacks):
        self.root = root
        self.app_callbacks = app_callbacks
        self.root.title("Sistema de Calibración Asistida")

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
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Configurar las columnas del frame principal
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # --- Columna Izquierda: Cámara y Barra de Comandos ---
        left_column_frame = ttk.Frame(main_frame)
        left_column_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_column_frame.rowconfigure(0, weight=1)

        camera_frame = ttk.LabelFrame(left_column_frame, text="Vista de Cámara", padding=5)
        camera_frame.grid(row=0, column=0, sticky="nsew")
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.pack(fill="both", expand=True)

        commands_frame = ttk.LabelFrame(left_column_frame, text="Barra de comandos", padding=10)
        commands_frame.grid(row=1, column=0, sticky="ew", pady=10)
        
        setpoint_frame = ttk.Frame(commands_frame)
        setpoint_frame.pack(fill='x', pady=2)
        ttk.Label(setpoint_frame, text="Setpoint CO2 (ppm):").pack(side='left')
        self.setpoint_entry = ttk.Entry(setpoint_frame, width=10)
        self.setpoint_entry.pack(side='left', padx=5)
        ttk.Button(setpoint_frame, text="Enviar", command=self.app_callbacks["send_setpoint"]).pack(side='left')

        pulse_frame = ttk.Frame(commands_frame)
        pulse_frame.pack(fill='x', pady=2)
        ttk.Label(pulse_frame, text="Pulso Válvula (ms):").pack(side='left')
        self.pulse_entry = ttk.Entry(pulse_frame, width=10)
        self.pulse_entry.pack(side='left', padx=5)
        ttk.Button(pulse_frame, text="Enviar", command=self.app_callbacks["send_pulse"]).pack(side='left')
        
        buttons_frame = ttk.Frame(commands_frame)
        buttons_frame.pack(fill='x', pady=(10, 2))
        ttk.Button(buttons_frame, text="Toggle Cooler", command=lambda: self.app_callbacks["send_command"]("TOGGLE_COOLER")).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(buttons_frame, text="Calibrar Sensor", command=lambda: self.app_callbacks["send_command"]("CALIBRATE_SENSOR")).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(buttons_frame, text="PÁNICO", command=lambda: self.app_callbacks["send_command"]("OPEN_ALL")).pack(side='left', expand=True, fill='x', padx=2)

        # --- Columna Derecha: Todo lo demás apilado ---
        right_column_frame = ttk.Frame(main_frame)
        right_column_frame.grid(row=0, column=1, sticky="nsew")
        right_column_frame.rowconfigure(3, weight=1) # Fila del gráfico se expande

        dashboard_frame = ttk.LabelFrame(right_column_frame, text="Dashboard", padding=10)
        dashboard_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        controls_ocr_frame = ttk.LabelFrame(right_column_frame, text="Controles OCR", padding=10)
        controls_ocr_frame.grid(row=1, column=0, sticky="ew", pady=5)

        debug_frame = ttk.LabelFrame(right_column_frame, text="Imágenes de depuración", padding=5)
        debug_frame.grid(row=2, column=0, sticky="ew", pady=5)
        
        plot_frame = ttk.LabelFrame(right_column_frame, text="Gráfico en tiempo real", padding=5)
        plot_frame.grid(row=3, column=0, sticky="nsew", pady=5)

        # === INICIO DEL CÓDIGO COMPLETADO ===

        # --- Llenar Dashboard ---
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

        # --- Llenar Controles OCR ---
        controls_ocr_frame.columnconfigure(1, weight=1)
        ttk.Label(controls_ocr_frame, text="Threshold:").grid(row=0, column=0, sticky='w')
        self.threshold_slider = ttk.Scale(
            controls_ocr_frame, from_=0, to=255, orient='horizontal',
            command=self.app_callbacks["on_threshold_change"]
        )
        self.threshold_slider.grid(row=0, column=1, sticky='ew', padx=5)
        
        roi_frame = ttk.Frame(controls_ocr_frame)
        roi_frame.grid(row=1, column=0, columnspan=2, pady=10)
        ttk.Label(roi_frame, text="Ajuste de ROI").grid(row=0, column=0, columnspan=3, pady=5)
        
        ttk.Button(roi_frame, text="↑", width=3, command=lambda: self.app_callbacks["adjust_roi"]('y', -5)).grid(row=1, column=1)
        ttk.Button(roi_frame, text="←", width=3, command=lambda: self.app_callbacks["adjust_roi"]('x', -5)).grid(row=2, column=0)
        ttk.Button(roi_frame, text="→", width=3, command=lambda: self.app_callbacks["adjust_roi"]('x', 5)).grid(row=2, column=2)
        ttk.Button(roi_frame, text="↓", width=3, command=lambda: self.app_callbacks["adjust_roi"]('y', 5)).grid(row=3, column=1)
        ttk.Button(roi_frame, text="W+", width=3, command=lambda: self.app_callbacks["adjust_roi"]('w', 5)).grid(row=4, column=0, pady=(10,0))
        ttk.Button(roi_frame, text="W-", width=3, command=lambda: self.app_callbacks["adjust_roi"]('w', -5)).grid(row=4, column=2, pady=(10,0))
        ttk.Button(roi_frame, text="H+", width=3, command=lambda: self.app_callbacks["adjust_roi"]('h', 5)).grid(row=5, column=0)
        ttk.Button(roi_frame, text="H-", width=3, command=lambda: self.app_callbacks["adjust_roi"]('h', -5)).grid(row=5, column=2)

        # --- Llenar Imágenes de Depuración ---
        debug_frame.columnconfigure((0, 1), weight=1)
        ttk.Label(debug_frame, text="Escala de Grises").grid(row=0, column=0)
        self.gray_label = ttk.Label(debug_frame)
        self.gray_label.grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Label(debug_frame, text="Threshold (Binarizada)").grid(row=0, column=1)
        self.thresh_label = ttk.Label(debug_frame)
        self.thresh_label.grid(row=1, column=1, padx=5, pady=5)

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
        
        # === FIN DEL CÓDIGO COMPLETADO ===

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
            self.thresh_label.image = imgtk_thresh
            self.thresh_label.configure(image=imgtk_thresh)
            
    def update_sensor_data(self, sensor_data, stable_reading):
        self.sensor_vars['TEMP'].set(f"{sensor_data.get('TEMP', '--.-')} °C")
        self.sensor_vars['HUM'].set(f"{sensor_data.get('HUM', '--.-')} %")
        self.sensor_vars['PRES'].set(f"{sensor_data.get('PRES', '----')} hPa")
        self.sensor_vars['CO2'].set(f"{sensor_data.get('CO2', '----')} ppm")
        self.sensor_vars['PCB1_STATE'].set(sensor_data.get('PCB1_STATE', 'UNKNOWN'))
        self.sensor_vars['PCB2_STATE'].set(sensor_data.get('PCB2_STATE', 'UNKNOWN'))
        self.sensor_vars['COOLER'].set(sensor_data.get('COOLER', 'UNKNOWN'))
        self.sensor_vars['OCR_STABLE'].set(f"{stable_reading} ppm")