import random
import time
from controller import Robot

# ---------------------
# Configuration robot
# ---------------------
TIME_STEP = 64
MAX_SPEED = 6.28
EPISODES = 500
STEPS_PER_EPISODE = 100
ALPHA = 0.1        # learning rate
GAMMA = 0.9        # discount factor
EPSILON = 0.9      # exploration factor
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1  # exploration minimale

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

# Q-table 8 états × 4 actions initialisée à petites valeurs positives
Q = [[0.1 for _ in range(4)] for _ in range(8)]

# ---------------------
# Fonctions d'action
# ---------------------
def forward():
    leftMotor.setVelocity(0.5 * MAX_SPEED)
    rightMotor.setVelocity(0.5 * MAX_SPEED)

def backward():
    leftMotor.setVelocity(-0.3 * MAX_SPEED)
    rightMotor.setVelocity(-0.3 * MAX_SPEED)

def left():
    leftMotor.setVelocity(0)           # roue gauche arrêtée pour virage serré
    rightMotor.setVelocity(0.5 * MAX_SPEED)  # roue droite avance vite

def right():
    leftMotor.setVelocity(0.5 * MAX_SPEED)   # roue gauche avance vite
    rightMotor.setVelocity(0)           # roue droite arrêtée

def stop():
    leftMotor.setVelocity(0)
    rightMotor.setVelocity(0)

def perform_action(action):
    if action == 0:
        forward()
    elif action == 1:
        left()
    elif action == 2:
        right()
    elif action == 3:
        backward()

# ---------------------
# Fonctions Q-learning
# ---------------------
def get_state():
    front = 1 if max(ps[0].getValue(), ps[1].getValue(), ps[7].getValue()) > 80 else 0
    left_sensor = 1 if max(ps[5].getValue(), ps[6].getValue()) > 80 else 0
    right_sensor = 1 if max(ps[2].getValue(), ps[3].getValue()) > 80 else 0
    return front * 4 + left_sensor * 2 + right_sensor

def choose_action(state):
    if random.random() < EPSILON:
        return random.randint(0, 3)
    return max(range(4), key=lambda a: Q[state][a])

def get_reward(action):
    threshold = 80
    front_hit = max(ps[0].getValue(), ps[1].getValue(), ps[7].getValue()) > threshold
    side_hit = max(ps[2].getValue(), ps[3].getValue(), ps[5].getValue(), ps[6].getValue()) > threshold

    if front_hit:
        return -50
    if side_hit:
        return -20
    if action == 0 and not front_hit and not side_hit:
        return 50
    if action in [1, 2] and not front_hit and not side_hit:
        return 5
    if action == 3:
        return 5
    return 10

# ---------------------
# Boucle d'apprentissage
# ---------------------
print("=== Début de l'apprentissage ===", flush=True)
for episode in range(EPISODES):
    state = get_state()
    print(f"\nÉpisode {episode+1}/{EPISODES}", flush=True)
    for step in range(STEPS_PER_EPISODE):
        action = choose_action(state)
        perform_action(action)
        robot.step(TIME_STEP)
        reward = get_reward(action)
        next_state = get_state()
        Q[state][action] = Q[state][action] + ALPHA * (reward + GAMMA*max(Q[next_state]) - Q[state][action])
        state = next_state
        time.sleep(0.01)
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

# ---------------------
# Phase de test améliorée
# ---------------------
print("\n=== Phase de test après apprentissage ===", flush=True)

TEST_DURATION = 60  # en secondes
test_steps = int((1000 / TIME_STEP) * TEST_DURATION)

seuil_obstacle = 80
obstacle_detected = False
turn_direction = None
turn_steps = 0
MAX_TURN_STEPS = 10

step = 0
while robot.step(TIME_STEP) != -1:
    front_vals = [ps[0].getValue(), ps[1].getValue(), ps[7].getValue()]
    left_vals = [ps[5].getValue(), ps[6].getValue()]
    right_vals = [ps[2].getValue(), ps[3].getValue()]

    max_front = max(front_vals)
    max_left = max(left_vals)
    max_right = max(right_vals)

    if max_front > seuil_obstacle or obstacle_detected:
        if not obstacle_detected:
            turn_direction = 'left' if max_left < max_right else 'right'
            obstacle_detected = True
            turn_steps = 0
            print(f"Obstacle détecté! Évitement vers {turn_direction}", flush=True)

        if turn_direction == 'left':
            left()
        else:
            right()

        turn_steps += 1

        if turn_steps >= MAX_TURN_STEPS:
            if max([ps[0].getValue(), ps[1].getValue(), ps[7].getValue()]) <= seuil_obstacle:
                obstacle_detected = False
                turn_direction = None
                print("Obstacle évité, reprise marche avant", flush=True)
                forward()
            else:
                turn_steps = 0
    else:
        # Pas d'obstacle : avancer droit
        forward()
        if step % 20 == 0:
            print(f"[TEST] Step {step}/{test_steps} | Avancer libre | Front: {[round(v,2) for v in front_vals]}", flush=True)

    step += 1
    if step >= test_steps:
        break

stop()
print("=== Fin du test ===", flush=True)

# ---------------------
# Sauvegarde Q-table finale
# ---------------------
with open("qtable.txt", "w") as f:
    for row in Q:
        f.write(f"{','.join(map(str,row))}\n")

print("\nQ-table finale sauvegardée dans qtable.txt", flush=True)
for i, row in enumerate(Q):
    print(f"État {i}: {row}", flush=True)
