# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 21:42:22 2025

@author: paco2
"""

import cv2

cap = cv2.VideoCapture(0)  # Cámara en tiempo real
x, y, w, h = 100, 100, 200, 200  # Valores iniciales de la ROI

while True:
    ret, frame = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1) & 0xFF
    
    # Selección manual de ROI con la tecla 'r'
    if key == ord('r'):
        roi = cv2.selectROI("Ajuste ROI", frame, showCrosshair=False)
        x, y, w, h = map(int, roi)
        print(x)
        print(y)
        print(w)
        print(h)
        cv2.destroyWindow("Ajuste ROI")
        
    print(x) 
    print(y)
    print(w)
    print(h)
    # Extraer y procesar ROI
    digit_roi = frame[y:y+h, x:x+w] if (w > 0 and h > 0) else frame
    # ... (aquí iría tu lógica de predicción)
    
    # Mostrar ROI y frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Ventana", frame)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()