#include <Servo.h>
#include <Stepper.h>

Servo servo01;
Servo servo02;
Servo servo03;
Servo servo04;
Servo servo05;
int speedDelay = 10;
int servo1SPos = 0;
int servo2SPos = 0;
int servo3SPos = 0;
int servo4SPos = 0;
int servo5SPos = 0;
int servo1EPos = 0;
int servo2EPos = 0;
int servo3EPos = 0;
int servo4EPos = 0;
int servo5EPos = 0;

const int stepsPerRevolution = 200; // Change this to fit the number of steps per revolution for your motor
Stepper myStepper(stepsPerRevolution, 9, 10, 11, 12);

void setup() {
  Serial.begin(9600); // Initialize serial communication at 9600 baud rate
  servo01.attach(7);
  servo02.attach(6);
  servo03.attach(5);
  servo04.attach(4);
  servo05.attach(3);
  myStepper.setSpeed(60);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    
    // Check if the command is for servos (e.g., "S [30 60 90 120 150]")
    if (command.startsWith("S ")) {
      command = command.substring(2, command.length() - 1); // Remove the "S " and the ending bracket
      
      int startIndex = 0;
      int valueIndex = 0;

      for (int i = 0; i <= command.length(); i++) {
        if (command.charAt(i) == ' ' || i == command.length()) {
          String valueStr = command.substring(startIndex, i);
          int value = valueStr.toInt();

          if (valueIndex == 0) {
            servo1EPos = value;
          } else if (valueIndex == 1) {
            servo2EPos = value;
          } else if (valueIndex == 2) {
            servo3EPos = value;
          } else if (valueIndex == 3) {
            servo4EPos = value;
          } else if (valueIndex == 4) {
            servo5EPos = value;
          }

          startIndex = i + 1;
          valueIndex++;
        }
      }

      for (int pos = servo1SPos; pos <= servo1EPos; pos++) {
        servo01.write(pos);
        delay(speedDelay);
      }
      servo1SPos = servo1EPos;

      for (int pos = servo2SPos; pos <= servo2EPos; pos++) {
        servo02.write(pos);
        delay(speedDelay);
      }
      servo2SPos = servo2EPos;

      for (int pos = servo3SPos; pos <= servo3EPos; pos++) {
        servo03.write(pos);
        delay(speedDelay);
      }
      servo3SPos = servo3EPos;

      for (int pos = servo4SPos; pos <= servo4EPos; pos++) {
        servo04.write(pos);
        delay(speedDelay);
      }
      servo4SPos = servo4EPos;

      for (int pos = servo5SPos; pos <= servo5EPos; pos++) {
        servo05.write(pos);
        delay(speedDelay);
      }
      servo5SPos = servo5EPos;
    }
    // Check if the command is for the stepper motor (e.g., "M 200")
    else if (command.startsWith("M ")) {
      command = command.substring(2);
      int steps = command.toInt();
      myStepper.step(steps);
      delay(500);
    }
  }
}
