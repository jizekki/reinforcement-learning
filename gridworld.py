import numpy as np

SIZE = 5
DISCOUNT = 0.9
# left, up, right, down
ACTIONS = [(0, -1), (-1, 0), (0, 1), (1, 0)]
EPSILON = 1e-4


def is_terminal(x, y):
    return (x == 0 and y == 0) or (x == SIZE - 1 and y == SIZE - 1)


def step(x, y, action):
    if is_terminal(x, y):
        return (x, y), 0
    next_x = x + ACTIONS[action][0]
    next_y = y + ACTIONS[action][1]
    if next_x < 0 or next_x >= SIZE or next_y < 0 or next_y >= SIZE:
        return (x, y), -1
    else:
        return (next_x, next_y), -1


def evaluate_random_policy(policy):
    value = np.zeros((SIZE, SIZE))
    while True:
        new_value = np.zeros_like(value)
        for i in range(SIZE):
            for j in range(SIZE):
                for action in range(len(ACTIONS)):
                    coordinates, recompense = step(i, j, action)
                    new_value[i, j] += policy[i, j, action] * \
                        (recompense + DISCOUNT *
                         value[coordinates[0], coordinates[1]])
        if np.sum(np.abs(value - new_value)) < EPSILON:
            return np.round(new_value, decimals=2)
        value = np.copy(new_value)


random_policy = np.full((SIZE, SIZE, len(ACTIONS)), 0.25)

print(evaluate_random_policy(random_policy))


def evaluate_deterministic_policy(policy):
    value = np.zeros((SIZE, SIZE))
    while True:
        new_value = np.zeros_like(value)
        for i in range(SIZE):
            for j in range(SIZE):
                coordinates, recompense = step(i, j, left_policy[i, j])
                new_value[i, j] += (recompense + DISCOUNT *
                                    value[coordinates[0], coordinates[1]])
        if np.sum(np.abs(value - new_value)) < EPSILON:
            return np.round(new_value, decimals=2)
        value = np.copy(new_value)


left_policy = {(i, j): 0 for i in range(SIZE) for j in range(SIZE)}

print(evaluate_deterministic_policy(left_policy))


def value_iteration():
    value = np.zeros((SIZE, SIZE))
    while True:
        new_value = np.zeros_like(value)
        for i in range(SIZE):
            for j in range(SIZE):
                tmp = []
                for action in range(len(ACTIONS)):
                    coordinates, recompense = step(i, j, action)
                    tmp.append(recompense + DISCOUNT *
                               value[coordinates[0], coordinates[1]])
                new_value[i, j] = np.max(tmp)
        if np.sum(np.abs(value - new_value)) < EPSILON:
            return np.round(new_value, decimals=2)
        value = np.copy(new_value)


print(value_iteration())


def policy_iteration():
    policy = left_policy
    stable = False
    while not stable:
        stable = True
        value = evaluate_deterministic_policy(policy)
        # define the improved strategy
        for i in range(SIZE):
            for j in range(SIZE):
                best_action = None
                best_value = -100
                for action in range(len(ACTIONS)):
                    coordinates, recompense = step(i, j, action)
                    val = recompense + DISCOUNT * \
                        value[coordinates[0], coordinates[1]]
                    if val > best_value:
                        best_value = val
                        best_action = action
                if best_action != policy[i, j]:
                    policy[i, j] = best_action
                    stable = False
    return value, policy


print(policy_iteration())
