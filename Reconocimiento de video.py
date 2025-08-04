import tensorflow as tf
import Funciones
import Modelo
import Ventanas
import cv2

# 
# opencv part
#

# if a model is already saved just load it - else build it
model = None

# the opencv display loop
def start_cv(model):
    
    #Definición de variables
    #print("\Definicion de variables")
    global threshold    # Crea variable global para aumentar el brillo
    cap = cv2.VideoCapture(0)   #Asigna la camara a cap
    res1 = 0

    frame = cv2.namedWindow('camara')   # Crea una ventana llamada camara
    
    #Inits
    print("\Inits")
    cv2.createTrackbar('threshold', 'camara', 150, 255, Funciones.on_threshold)   #Crea la barra para variar el brillo
    
    #Crea el menu
    print("Creo el menu")
    menu_image = Ventanas.create_menu()
    cv2.imshow('Menu', menu_image)
    print("Entro al bucle")
    
    
    #brillo = 0
    bDigito = 50
    hDigito = 100
    
    a = 200
    b = 250
    
    # a = Punto inicial del 1er digito
    # h1digito = Altura del 1er digito
    # b -> Punto inicial de la altura
    # 
    
    #a:a+hDigito, b+bDigito:b+2*bDigito   #Captura una zona de la camara
        
    while True: #Bucle infinito
        key = cv2.waitKey(1) & 0xff
        ret, frame = cap.read() #ret indica si se puede capturar el frame o no
    
        if key == ord('5'):
            a = 100
            b = 200
        
        if key == ord('6'):
            a = 200
            b = 250
            
        #Test de area de selección de imagen
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # pasa a escala de grises lo capturado por la camara
        _, thr = cv2.threshold(grayFrame, Funciones.threshold, 255, cv2.THRESH_BINARY_INV)  # aplica el thershold y guarda en thr 
        #Region de interes 1er digito
        resizedFrame1 = thr[a:a+hDigito, b+bDigito:b+2*bDigito]   #Captura una zona de la camara
        #Detección de un digito adicional
        resizedFrame1_color = cv2.cvtColor(resizedFrame1, cv2.COLOR_GRAY2BGR)   #Pasa de escala de grises a BGR para poder insentar en la imagen de la camara
        frame[a:a+hDigito, b:b+bDigito] = resizedFrame1_color
        cv2.rectangle(frame, (b, a), (b+bDigito, a+hDigito), (0, 0, 255), thickness=1) #Este rectangulo encierra la parte del reconocimiento
             
        
        #DETECTAR SI SON 3 o 4 digitos - Cambia la zona donde hago la predicción    
        
        #Manejo de la camara
       # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # pasa a escala de grises lo capturado por la camara
       # _, thr = cv2.threshold(grayFrame, Funciones.threshold, 255, cv2.THRESH_BINARY_INV)  # aplica el thershold y guarda en thr 
        #Region de interes 1er digito
       # resizedFrame1 = thr[200:200+hDigito, 250+bDigito:250+2*bDigito]   #Captura una zona de la camara
        #Detección de un digito adicional
       # resizedFrame1_color = cv2.cvtColor(resizedFrame1, cv2.COLOR_GRAY2BGR)   #Pasa de escala de grises a BGR para poder insentar en la imagen de la camara
       # frame[200:200+hDigito, 250:250+bDigito] = resizedFrame1_color
       # cv2.rectangle(frame, (250, 200), (250+bDigito, 200+hDigito), (0, 0, 255), thickness=1) #Este rectangulo encierra la parte del reconocimiento

  
        #Predicción de digito
        iconImg1 = cv2.resize(resizedFrame1, (28, 28)) # redimenciona la captura para pasarla al modelo
        res1 = Funciones.predict(model, iconImg1) #Pasa el recorte al modelo
        
        #actualiza el menu
        updated_menu = menu_image.copy()
        cv2.putText(updated_menu, f"{res1}", (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        #Muestra las imagenes
        cv2.imshow('Menu', updated_menu)    #Muestra el menu
        cv2.imshow('camara', frame)         #muestra la camara normal
 
        #Opciones
        if key == ord('1'):
            print("Muestro el menu de Calibración")
        elif key == ord('2'):
            print("Muestro el menu de conecntraciones")
        elif key == ord('q'):
            cv2.destroyWindow('camara') #Destruye la ventana background
            break   #Sale del bucle
           
    print ("Salgo del bucle, vacio recursos de la camara y destruyo todas las pantallas")   
    cap.release()   # Borra los recursos utilizados por la camara
    cv2.destroyAllWindows() # destruye todas las ventanas creadas
        
# main function 
def main():
    #Inicialización del serial
   
    #Comunicación serial con el PCB2
    #menu()
    #LOG
    #Comparación
    
    # Simula los datos que vendrían de tus sensores
    datos_actuales = {
        'co2_gm70': 3,
        'co2_mhz19c': 455,
        'presion': 1012.5,
        'temperatura': 24.8,
        'humedad': 65,
        'bateria': 87
    }
    
    # Crea la imagen del menú con los datos
    menu_visual = Ventanas.crear_menu_mejorado(datos_actuales)
    
    # Muestra la imagen
    cv2.imshow("Menu Mejorado", menu_visual)
    
    try:
        model = tf.keras.models.load_model('model.sav') #Crea el modelo 
        print('loaded saved model.')
        print(model.summary())
    except:
       # load and train data 
        print("getting mnist data...")
        #(x_train, y_train, x_test, y_test) = get_mnist_data()
        (x_train, y_train, x_test, y_test) = Modelo.get_mnist_data()
        print("training model...")
        #model = train_model(x_train, y_train, x_test, y_test)
        model = Modelo.train_model(x_train, y_train, x_test, y_test)
        print("saving model...")
        model.save('my_model.keras')
        #model.save('model.h5')
        print("Model saved")
    
    #print("starting cv...")
    print("Inicialización de openCV")
    #show_menu()
    # show opencv window
    start_cv(model)
    #start_cv(model)

# call main
if __name__ == '__main__':
    main()