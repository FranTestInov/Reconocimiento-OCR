# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 00:26:10 2025

@author: paco2
"""

# app/ui_manager.py
import cv2
import numpy as np

class UIManager:
    def __init__(self, window_names):
        self.cam_window = window_names['camera']
        self.menu_window = window_names['menu']
        self.menu_image_base = self._create_menu_image()

    def setup_windows(self, initial_threshold, threshold_callback):
        """Crea las ventanas y el trackbar."""
        cv2.namedWindow(self.cam_window)
        cv2.createTrackbar('Threshold', self.cam_window, initial_threshold, 255, threshold_callback)
        cv2.imshow(self.menu_window, self.menu_image_base)

    def _create_menu_image(self):
        """Crea la imagen de fondo para el menú."""
        menu = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(menu, "Sistema de Calibracion Asistida", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.line(menu, (30, 70), (610, 70), (255, 255, 255), 1)
        cv2.putText(menu, "Lectura OCR", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        cv2.putText(menu, "Datos de Sensores (ESP32)", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        return menu

    def update_display(self, frame, sensor_data, stable_reading, debug_images=None):
        """Actualiza ambas ventanas con los datos más recientes."""
        # Actualizar menú
        updated_menu = self.menu_image_base.copy()
        cv2.putText(updated_menu, stable_reading, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Datos de Sensores y Estados
        cv2.putText(updated_menu, f"Temp: {sensor_data['TEMP']} C", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(updated_menu, f"Hum: {sensor_data['HUM']} %", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(updated_menu, f"Pres: {sensor_data['PRES']} hPa", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(updated_menu, f"CO2: {sensor_data['CO2']} ppm", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(updated_menu, f"PCB1: {sensor_data['PCB1_STATE']}", (350, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
        cv2.putText(updated_menu, f"Cooler: {sensor_data['COOLER']}", (350, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
        cv2.putText(updated_menu, f"PCB2: {sensor_data['STATUS']}", (350, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
        
        cv2.imshow(self.menu_window, updated_menu)

        # Actualizar frame de la cámara con depuración
        if debug_images:
            gray_roi, thr_roi = debug_images
            self._draw_debug_windows(frame, gray_roi, thr_roi)

        cv2.imshow(self.cam_window, frame)

    def _draw_debug_windows(self, frame, gray_roi, thr_roi):
        """Dibuja las pequeñas ventanas de depuración en el frame principal."""
        debug_w, debug_h = 160, 80
        gray_debug = cv2.resize(gray_roi, (debug_w, debug_h))
        thr_debug = cv2.resize(thr_roi, (debug_w, debug_h))
        gray_bgr = cv2.cvtColor(gray_debug, cv2.COLOR_GRAY2BGR)
        thr_bgr = cv2.cvtColor(thr_debug, cv2.COLOR_GRAY2BGR)

        frame_h, frame_w, _ = frame.shape
        y_start = frame_h - debug_h - 10
        x_start1 = frame_w - (debug_w * 2) - 15
        x_start2 = frame_w - debug_w - 10
        
        frame[y_start:y_start+debug_h, x_start1:x_start1+debug_w] = gray_bgr
        frame[y_start:y_start+debug_h, x_start2:x_start2+debug_w] = thr_bgr
        cv2.putText(frame, "Grises", (x_start1, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Threshold", (x_start2, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    @staticmethod
    def close_all_windows():
        cv2.destroyAllWindows()