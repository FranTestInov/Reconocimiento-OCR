// --- CONFIGURACIÓN DE PINES ---
// Definí los pines GPIO donde conectaste los relés de las válvulas
const int PIN_VALVULA_1 = 2; // Ejemplo, usá el pin que corresponda
const int PIN_VALVULA_2 = 4; // Ejemplo

// --- VARIABLES PARA EL ENVÍO PERIÓDICO DE DATOS ---
unsigned long previousMillis = 0;       // Almacenará el tiempo del último envío
const long interval = 2000;             // Intervalo de envío en milisegundos (2 segundos)

// --- SETUP ---
// Se ejecuta una sola vez al encender el ESP32
void setup() {
  // Inicia la comunicación serial a la misma velocidad que en Python
  Serial.begin(115200);
  
  // Configura los pines de las válvulas como salidas
  pinMode(PIN_VALVULA_1, OUTPUT);
  pinMode(PIN_VALVULA_2, OUTPUT);
  
  // Asegurarse de que las válvulas empiecen apagadas
  digitalWrite(PIN_VALVULA_1, LOW);
  digitalWrite(PIN_VALVULA_2, LOW);
  
  // Mensaje de inicio
  Serial.println("ESP32 listo para recibir comandos.");
}

// --- LOOP ---
// Se ejecuta constantemente
void loop() {
  // Revisa si hay algún dato esperando en el buffer serial
  if (Serial.available() > 0) {
    // Lee el comando completo hasta que encuentra un salto de línea ('\n')
    String command = Serial.readStringUntil('\n');
    command.trim(); // Limpia espacios en blanco o caracteres invisibles
    
    // Procesa el comando recibido
    if (command == "V1_ON") {
      digitalWrite(PIN_VALVULA_1, HIGH); // Enciende el relé
      Serial.println("OK: Valvula 1 ENCENDIDA"); // Envía confirmación a la PC
    }
    else if (command == "V1_OFF") {
      digitalWrite(PIN_VALVULA_1, LOW); // Apaga el relé
      Serial.println("OK: Valvula 1 APAGADA");
    }
    // Podrías añadir más comandos para la Válvula 2 de la misma forma
    // else if (command == "V2_ON") { ... }
    else {
      // Si el comando no se reconoce
      Serial.println("ERROR: Comando no reconocido");
    }
  }
}