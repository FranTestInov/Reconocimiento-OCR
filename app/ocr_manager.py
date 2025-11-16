# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 00:25:51 2025

@author: paco2
"""

# app/ocr_manager.py
import cv2
import pytesseract
import logging
from collections import Counter

class OCRManager:
    def __init__(self, config):
        self.config = config
        self.tesseract_path = config['tesseract']['command_path']
        self._initialize_tesseract()

        self.readings_buffer = []
        self.stable_reading = "---"

    def _initialize_tesseract(self):
        try:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            logging.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logging.error(f"No se pudo encontrar Tesseract en '{self.tesseract_path}'. Error: {e}")

    def process_frame(self, frame, roi_coords, threshold_value):
        """Realiza el OCR sobre una ROI del frame y actualiza el búfer."""
        x, y, w, h = roi_coords
        if w <= 0 or h <= 0:
            return None

        roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thr_roi = cv2.threshold(gray_roi, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]

        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        raw_text = pytesseract.image_to_string(thr_roi, config=config).strip()
        
        logging.debug(f"Tesseract leyó: '{raw_text}'")
        validated_text = self._validate_reading(raw_text)

        if validated_text:
            self.readings_buffer.append(validated_text)
            buffer_size = self.config['detection']['validation_buffer']['size']
            if len(self.readings_buffer) > buffer_size:
                self.readings_buffer.pop(0)
        
        return gray_roi, thr_roi

    def _validate_reading(self, text):
        """Aplica reglas de validación a la lectura del OCR."""
        if not text.isdigit(): return None
        if not (len(text) == 3 or len(text) == 4): return None
        if not text.endswith('0'): return None
        #if not text < text+1500: return None
        return text

    def update_stable_reading(self):
        """
        Analiza el búfer y determina si hay una nueva lectura estable.
        Configura el buffer segun el archivo de configuración
        """
        buffer_conf = self.config['detection']['validation_buffer']
        buffer_size = len(self.readings_buffer)

        if buffer_size < buffer_conf['size'] / 2:
            return False

        counts = Counter(self.readings_buffer)
        if not counts: return False

        candidate, count = counts.most_common(1)[0]
        confidence = count / buffer_size

        if confidence >= buffer_conf['confidence_threshold'] and self.stable_reading != candidate:
            self.stable_reading = candidate
            logging.info(f"NUEVO VALOR ESTABLE: '{candidate}' (confianza: {confidence:.0%})")
            return True
        return False