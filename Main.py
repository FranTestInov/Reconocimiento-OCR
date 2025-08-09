# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 19:23:11 2025

1- Imports: Todas las librerías necesarias al principio.
2- Constantes y Configuración: Parámetros que no cambian, como nombres de 
ventanas, rutas de archivos o configuraciones iniciales.
3- Definición de Clases: El corazón de la aplicación. Aquí definiremos nuestra 
clase principal que manejará la cámara, el modelo y la interfaz.
4- Función main: Una función principal corta y clara que crea una instancia de 
nuestra clase y la ejecuta.
5- Punto de Entrada: El bloque if __name__ == '__main__': que llama a la 
función main.
@author: paco2
"""

# 1. IMPORTS
import tensorflow as tf
import cv2
import numpy as np
import os
# Asumimos que los otros archivos (Modelo, Funciones, Ventanas) se usarán o su lógica se integrará aquí.
# Por ahora, para simplificar, integraré la lógica de Funciones y Ventanas directamente.

# 2. CONFIGURACIÓN
class Config:
    """Una clase para mantener todas las configuraciones en un solo lugar."""
    WINDOW_NAME_CAM = 'Camara'
    WINDOW_NAME_MENU = 'Menu'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'my_model.keras')
    #MODEL_PATH = 'my_model.keras'
    BIG_ROI_INITIAL_X = 150
    BIG_ROI_INITIAL_Y = 240
    BIG_ROI_INITIAL_W = 200
    BIG_ROI_INITIAL_H = 120
    PREDICTION_IMG_SIZE = (28, 28)

# 3. CLASE PRINCIPAL DE LA APLICACIÓN
class CalibratorApp:
    """
    Clase principal que encapsula toda la lógica de la aplicación de calibración.
    """
    def __init__(self, config):
        """
        Constructor: Inicializa todos los componentes de la aplicación.
        """
        print("Construye la configuración")
        self.config = config
        print("Construye el modelo")
        self.model = self._initialize_model()
        
        #Abre la camara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("No se puede abrir la cámara")
            
        # Estado de la aplicación
        self.threshold_value = 150
        self.big_roi_x = self.config.BIG_ROI_INITIAL_X
        self.big_roi_y = self.config.BIG_ROI_INITIAL_Y
        self.big_roi_w = self.config.BIG_ROI_INITIAL_W
        self.big_roi_h = self.config.BIG_ROI_INITIAL_H
        
        # Inicialización de la UI
        self._setup_ui()

    def _initialize_model(self):
        """
        Intenta cargar un modelo pre-entrenado. Si no lo encuentra,
        dispara el proceso de entrenamiento.
        """
        try:
            print(f"Intentando cargar modelo desde: {self.config.MODEL_PATH}")
            model = tf.keras.models.load_model(self.config.MODEL_PATH)
            #model = tf.keras.models.load_model('my_model.keras')
            print("¡Modelo cargado exitosamente desde archivo!")
            model.summary()
            return model
        except (IOError, OSError):# Captura errores de archivo no encontrado
            print(f"Modelo no encontrado en {self.config.MODEL_PATH}.")
            print("Se procederá a crear y entrenar un nuevo modelo.")
            return self._create_and_train_new_model()
        
        def _create_and_train_new_model(self):  
            
            """
            Encapsula toda la lógica de obtención de datos y entrenamiento
            que tenías en Modelo.py.
            """
            # 1. Obtener los datos (get_mnist_data)
            print("Obteniendo datos de MNIST...")
            #(x_train, y_train, x_test, y_test) = Modelo.get_mnist_data()
           
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')
            x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalización
            
            # 2. Definir el modelo (parte de train_model)
            model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])  
            
            model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
            print("\n--- Resumen del Nuevo Modelo ---")
            model.summary()
            print("---------------------------------\n")

            # 3. Entrenar el modelo (parte de train_model)
            print("Iniciando entrenamiento...")
            
            # Defino el callback para detener el entrenamiento temprano
            class MyCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    # CORRECCIÓN: Detenerse al 99% de accuracy, no al 10%
                    if logs.get('accuracy') > 0.99:
                        print("\n¡Se alcanzó 99% de accuracy, cancelando entrenamiento!")
                        self.model.stop_training = True
            
            history = model.fit(x_train, y_train, epochs=10, callbacks=[MyCallback()])
            print(f"Entrenamiento finalizado. Accuracy final: {history.history['accuracy'][-1]:.4f}")
            
            # 4. Guardar el modelo para la próxima vez
            print(f"Guardando el nuevo modelo en: {self.config.MODEL_PATH}")
            model.save(self.config.MODEL_PATH)
            
            return model




    def _setup_ui(self):
        """Configura las ventanas y los controles de OpenCV."""
        cv2.namedWindow(self.config.WINDOW_NAME_CAM)
        cv2.createTrackbar('Threshold', self.config.WINDOW_NAME_CAM, self.threshold_value, 255, self._on_threshold_change)
        
        self.menu_image = self._create_menu_image()
        cv2.imshow(self.config.WINDOW_NAME_MENU, self.menu_image)
        
    def run(self):
        """
        Bucle principal de la aplicación.
        """
        if self.model is None:
            print("Saliendo de la aplicación porque no se pudo cargar el modelo.")
            self.cleanup()
            return

        print("Iniciando bucle principal...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error al capturar el frame. Saliendo.")
                break

            # Procesar la entrada del teclado
            key = cv2.waitKey(1) & 0xFF
            if self._handle_keyboard_input(key):
                break # Salir si se presiona 'q'

            # Procesar el frame y realizar la predicción
            predicted_value  = self._process_frame (frame)

            # Actualizar y mostrar las ventanas
            self._update_display(frame, predicted_value)

        # Limpiar recursos al salir del bucle
        self.cleanup()
        
    def _process_frame (self, frame):
        """Encuentra, ordena y predice dígitos usando la ROI dinámica de la instancia."""
 
        # 1. Usa los atributos de la instancia (self) en lugar de variables locales
        x, y, w, h = self.big_roi_x, self.big_roi_y, self.big_roi_w, self.big_roi_h
         
        # Dibuja el rectángulo de la ROI grande
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
         
        # Recorta la ROI grande del frame original
        roi_grande = frame[y:y+h, x:x+w]
         
        # Procesa la ROI
        gray_roi = cv2.cvtColor(roi_grande, cv2.COLOR_BGR2GRAY)
        _, thr_roi = cv2.threshold(gray_roi, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
         
        # --- El resto de la lógica de findContours es la misma ---
        contours, _ = cv2.findContours(thr_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         
        # Filtrar y ordenar contornos
        filtered_contours = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
        sorted_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        predicted_digits = []
        for c in sorted_contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            digit_roi = thr_roi[cy:cy+ch, cx:cx+cw]
             
            # Dibuja el rectángulo del dígito encontrado (coordenadas relativas a la ventana principal)
            cv2.rectangle(frame, (x + cx, y + cy), (x + cx + cw, y + cy + ch), (0, 255, 0), 2)
             
            if digit_roi.size > 0:
                preprocessed_img = cv2.resize(digit_roi, self.config.PREDICTION_IMG_SIZE)
                imgs_to_predict = np.array([preprocessed_img])
                prediction_result = self.model.predict(imgs_to_predict, verbose=0)
                predicted_digit = str(np.argmax(prediction_result))
                predicted_digits.append(predicted_digit)
                 
        final_number = "".join(predicted_digits) if predicted_digits else "N/A"
        return final_number    

    def _handle_keyboard_input(self, key):
        """Maneja las teclas presionadas por el usuario. Devuelve True si se debe salir."""
        if key == ord('q'):
            return True
        elif key == ord('1'):
            print("Opción 1: Calibrar (lógica no implementada)")
        elif key == ord('2'):
            print("Opción 2: Definir concentración (lógica no implementada)")
        # Ejemplo de cómo cambiar la ROI dinámicamente
        # elif key == ord('j'):
        #     self.roi_x -= 5
        # elif key == ord('l'):
        #     self.roi_x += 5
        # elif key == ord('i'):
        #     self.roi_y -= 5
        # elif key == ord('k'):
        #     self.roi_y += 5
            
        # Mover la ROI
        if key == ord('w'): self.big_roi_y -= 5
        if key == ord('s'): self.big_roi_y += 5
        if key == ord('a'): self.big_roi_x -= 5
        if key == ord('d'): self.big_roi_x += 5
        
        # Redimensionar la ROI
        if key == ord('e'): self.big_roi_w += 5 # Agrandar ancho
        if key == ord('r'): self.big_roi_h += 5 # Agrandar alto
        if key == ord('c'): self.big_roi_w -= 5 # Achicar ancho
        if key == ord('v'): self.big_roi_h -= 5 # Achicar alto
            
        return False

    def _update_display(self, frame, predicted_digit):
        """Actualiza el menú con la nueva predicción y refresca las ventanas."""
        # Crea una copia del menú para no dibujar sobre la imagen original
        updated_menu = self.menu_image.copy()
        cv2.putText(updated_menu, predicted_digit, (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        cv2.imshow(self.config.WINDOW_NAME_MENU, updated_menu)
        cv2.imshow(self.config.WINDOW_NAME_CAM, frame)

    def cleanup(self):
        """Libera los recursos de la cámara y destruye las ventanas."""
        print("Limpiando recursos y cerrando aplicación.")
        self.cap.release()
        cv2.destroyAllWindows()

    # --- Funciones de ayuda / Callbacks ---
    def _on_threshold_change(self, value):
        """Callback para la barra de threshold."""
        self.threshold_value = value
        
    def _create_menu_image(self):
        """Crea la imagen base para el menú de la UI."""
        # Esta función es la que tenías en Ventanas.py
        menu = np.zeros((400, 500, 3), np.uint8)
        cv2.putText(menu, "Menu Principal", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(menu, "Use 'i,j,k,l' para mover la ROI", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(menu, "Presione 'q' para salir", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(menu, "Prediccion (Vaisala):", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return menu

# 4. FUNCIÓN MAIN
def main():
    """
    Función principal que instancia y ejecuta la aplicación.
    """
    app_config = Config()
    app = CalibratorApp(app_config)
    app.run()

# 5. PUNTO DE ENTRADA
if __name__ == '__main__':
    main()