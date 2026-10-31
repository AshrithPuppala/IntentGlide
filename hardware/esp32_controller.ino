/*
 * IntentGlide ESP32 ERM vibration-rig starter firmware.
 *
 * IMPORTANT:
 * - Do not connect an ERM motor directly to an ESP32 GPIO.
 * - Use a suitable logic-level MOSFET / motor driver and appropriate motor supply.
 * - Connect grounds together where required by the driver topology.
 * - This is a controlled engineering disturbance, not a clinical tremor model.
 * - Confirm PWM API/pin assignments for your specific ESP32 board/core version.
 */

const int MOTOR_PWM_PIN = 25;
const int PWM_FREQUENCY = 20000;
const int PWM_RESOLUTION_BITS = 8;
const int DEFAULT_DUTY = 110;  // 0..255; tune after checking motor current
const unsigned long ON_DURATION_MS = 800;
const unsigned long OFF_DURATION_MS = 700;

void setup() {
  Serial.begin(115200);
  pinMode(MOTOR_PWM_PIN, OUTPUT);
  analogWriteResolution(PWM_RESOLUTION_BITS);
  analogWriteFrequency(MOTOR_PWM_PIN, PWM_FREQUENCY);
  analogWrite(MOTOR_PWM_PIN, 0);
  Serial.println("IntentGlide vibration rig ready. Send '1' to pulse, '0' to stop.");
}

void loop() {
  if (Serial.available()) {
    const char command = Serial.read();
    if (command == '1') {
      analogWrite(MOTOR_PWM_PIN, DEFAULT_DUTY);
      delay(ON_DURATION_MS);
      analogWrite(MOTOR_PWM_PIN, 0);
      delay(OFF_DURATION_MS);
      Serial.println("pulse_complete");
    } else if (command == '0') {
      analogWrite(MOTOR_PWM_PIN, 0);
      Serial.println("motor_stopped");
    }
  }
}
