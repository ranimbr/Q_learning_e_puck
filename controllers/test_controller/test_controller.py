import time
import random
from controller import Robot

# ---------------------
# Configuration robot
# ---------------------
TIME_STEP = 64
MAX_SPEED = 6.28
SEUIL_OBSTACLE = 80

robot = Robot()

# Moteurs
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0)
rightMotor.setVelocity(0)

# Capteurs IR
psNames = ['ps0','ps1','ps2','ps3','ps4','ps5','ps6','ps7']
ps = [robot.getDevice(name) for name in psNames]
for sensor in ps:
    sensor.enable(TIME_STEP)

# ---------------------
# Charger Q-table
# ---------------------
Q = []
with open("qtable.txt", "r") as f:
    for line in f:
        Q.append([float(x) for x in line.strip().split(",")])

print("Q-table chargée :")
for i, row in enumerate(Q):
    print(f"État {i}: {row}")

# ---------------------
# Actions
# ---------------------
def forward():
    leftMotor.setVelocity(0.5 * MAX_SPEED)
    rightMotor.setVelocity(0.5 * MAX_SPEED)

def left():
    leftMotor.setVelocity(0)
    rightMotor.setVelocity(0.5 * MAX_SPEED)

def right():
    leftMotor.setVelocity(0.5 * MAX_SPEED)
    rightMotor.setVelocity(0)

def backward():
    leftMotor.setVelocity(-0.3 * MAX_SPEED)
    rightMotor.setVelocity(-0.3 * MAX_SPEED)

def perform_action(action):
    if action == 0: forward()
    elif action == 1: left()
    elif action == 2: right()
    elif action == 3: backward()

# ---------------------
# États (identique au training)
# ---------------------
def get_state():
    front = 1 if max(ps[0].getValue(), ps[1].getValue(), ps[7].getValue()) > 80 else 0
    left_s = 1 if max(ps[5].getValue(), ps[6].getValue()) > 80 else 0
    right_s = 1 if max(ps[2].getValue(), ps[3].getValue()) > 80 else 0
    return front * 4 + left_s * 2 + right_s

# ---------------------
# Phase de test (seul contrôleur ici)
# ---------------------
print("=== Contrôleur de test basé sur la Q-table ===")

obstacle_detected = False
turn_direction = None
turn_steps = 0
MAX_TURN_STEPS = 10

step = 0
while robot.step(TIME_STEP) != -1:

    # Lire capteurs
    front_vals = [ps[0].getValue(), ps[1].getValue(), ps[7].getValue()]
    left_vals = [ps[5].getValue(), ps[6].getValue()]
    right_vals = [ps[2].getValue(), ps[3].getValue()]

    max_front = max(front_vals)
    max_left = max(left_vals)
    max_right = max(right_vals)

    # Détection obstacle
    if max_front > SEUIL_OBSTACLE or obstacle_detected:

        if not obstacle_detected:
            turn_direction = 'left' if max_left < max_right else 'right'
            obstacle_detected = True
            turn_steps = 0
            print(f"Obstacle détecté → Évitement vers {turn_direction}")

        if turn_direction == 'left':
            left()
        else:
            right()

        turn_steps += 1

        if turn_steps >= MAX_TURN_STEPS:
            if max([ps[0].getValue(), ps[1].getValue(), ps[7].getValue()]) <= SEUIL_OBSTACLE:
                obstacle_detected = False
                print("Obstacle évité → Reprise avant")
                forward()
            else:
                turn_steps = 0

    else:
        # Mode Q-learning pur
        state = get_state()
        action = max(range(4), key=lambda a: Q[state][a])
        perform_action(action)

        if step % 20 == 0:
            print(f"[TEST] step={step} | état={state} | action={action} | front={front_vals}")

    step += 1

# arrêt
leftMotor.setVelocity(0)
rightMotor.setVelocity(0)
print("=== Fin du test ===")
