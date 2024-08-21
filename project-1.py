import cv2

metodo_deteccion = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializa la cámara (0 es normalmente la cámara predeterminada)
camara = cv2.VideoCapture(0)

while True:
    status, frame = camara.read()

    imagen_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostros en la imagen
    rostros = metodo_deteccion.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #print("Encontrados {0} rostros!".format(len(rostros)))
    # La variable rostros contiene 4 valores: x, y, ancho y alto de cada rostro detectado
    # primer valor es la coordenada x, segundo valor es la coordenada y, tercer valor es el ancho, cuarto valor es el alto
    #print(rostros)
    

    # Recorrer la lista para obtener los valores individuales
    for (x, y, largo, alto) in rostros:
        # Dibuja un rectángulo alrededor de los rostros
        # En donde se va a dibujar el rectángulo
        # La posición en x y del centro del rectangulo
        # El largo y ancho del rectangulo
        # El color del rectangulo en BGR (blue, green, red)
        # El grosor del rectangulo
        cv2.rectangle(frame, (x, y), (x+largo, y+alto), (230, 66, 245), 2)
        # imprimir un punto rojo en el centro del rectangulo
        cv2.circle(frame, (x + int(largo/2), y + int(alto/2)), 5, (0, 0, 255), -1)
         
        print(f'''
        La posición en X es: {x}
        La posición en Y es: {y}
        El alto del rostro es: {alto}
        El largo del rostro es: {largo}
        ''')

    # Recorrer la lista para obtener los valores individuales
    #for (x, y, w, h) in rostros:
        # Dibuja un rectángulo alrededor de los rostros
       # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    


    # Si no se puede leer el frame, sal del bucle
    if not status:
        break

    cv2.imshow('Video', frame)

    # Espera 1 ms para ver si se presiona la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
camara.release()
cv2.destroyAllWindows()
