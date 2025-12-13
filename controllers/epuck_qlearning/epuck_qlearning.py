from controller import Robot
import numpy as np
import random

TIME_STEP = 64

robot = Robot()

# ------- E-PUCK -------
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0)
right_motor.setVelocity(0)

# ------- capteurs -------
ps_names = [
    "ps0","ps1","ps2","ps3",
    "ps4","ps5","ps6","ps7"
]

ps = []
for name in ps_names:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    ps.append(sensor)

# ------- Q-learning -------
n_states = 4              # catégories d’état
n_actions = 3             # actions

Q = np.zeros((n_states, n_actions))

alpha = 0.1               # learning rate
gamma = 0.9               # discount
epsilon = 0.1             # exploration

# ------- discretisation -------
def get_state():
    # capteurs de gauche/droite
    left = ps[0].getValue() + ps[1].getValue() + ps[2].getValue()
    right = ps[5].getValue() + ps[6].getValue() + ps[7].getValue()

    # simple discretisation
    if left < 100 and right < 100:
        return 0   # libre
    if left > right:
        return 1   # obstacle gauche
    if right > left:
        return 2   # obstacle droite
    return 3       # inconnu


# ------- actions moteur -------
def do_action(a):
    if a == 0:    # avancer
        left_motor.setVelocity(5.0)
        right_motor.setVelocity(5.0)

    elif a == 1:  # tourner gauche
        left_motor.setVelocity(2.0)
        right_motor.setVelocity(5.0)

    elif a == 2:  # tourner droite
        left_motor.setVelocity(5.0)
        right_motor.setVelocity(2.0)


# ------- récompense -------
def compute_reward():
    left = ps[0].getValue() + ps[1].getValue() + ps[2].getValue()
    right = ps[5].getValue() + ps[6].getValue() + ps[7].getValue()

    # si obstacles -> punir
    if left > 250 or right > 250:
        return -1

    # sinon, avancer = +1
    return 1


# ------- boucle principale -------
while robot.step(TIME_STEP) != -1:

    s = get_state()

    # exploration OR exploitation
    if random.random() < epsilon:
        a = random.randint(0, n_actions - 1)
    else:
        a = np.argmax(Q[s])

    do_action(a)

    r = compute_reward()
    s2 = get_state()

    # mise à jour Q
    Q[s][a] = Q[s][a] + alpha * (
        r + gamma * np.max(Q[s2]) - Q[s][a]
    )

    # affichage simple
    print("state:", s, "action:", a, "reward:", r)
    print("Q:", Q)
    print("------------------")
