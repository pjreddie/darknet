
import maestro


servo = maestro.Controller()

#control the pitch, 5000 down, 8000 up
servo.setTarget(0, 6000) 

#control the yaw, 5000 left, 7000 right
servo.setTarget(1, 5000) 
