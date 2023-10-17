import csv
import numpy as np
import math
import matplotlib.pyplot as plt


def get_dist_matrix(path: str):
    coords, costs = get_coords_n_costs(path)

    distances = np.round(np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))).astype(float)
    distances[distances == 0] = np.inf

    return distances + costs


def get_coords_n_costs(path: str):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)

    data = np.array(data).astype(int)

    coords = data[:, :2]
    costs = data[:, 2]

    return coords, costs


def random_solution(matrix):
    n = math.ceil(matrix.shape[0] / 2)

    # I wonder whether it should be even simpler or it is ok to leave it in this form
    sol = np.array(np.random.choice(matrix.shape[0], size=n, replace=False))

    cost = calculate_cost(sol, matrix)

    return sol, cost


def calculate_cost(solution, matrix):
    return sum(matrix[solution[i-1], solution[i]] for i in range(len(solution)))


def show_solution(path, solution, title):
    coords, node_costs = get_coords_n_costs(path)
    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c=node_costs, cmap='plasma')

    best_tour_coords = np.append(solution, solution[0])
    plt.plot(coords[best_tour_coords, 0], coords[best_tour_coords, 1], 'r-')

    plt.colorbar(label='Cost')
    plt.title(title)
    plt.show()


def run_random_exp(path: str):
    matrix = get_dist_matrix(path)

    solutions = []

    for _ in range(200):
        solutions.append(random_solution(matrix))

    costs = np.array([cost for sol, cost in solutions])
    best_sol, best_cost = min(solutions, key=lambda x: x[1])
    worst_cost = np.max(costs)
    avg_cost = np.mean(costs)

    print("Best cost: " + str(best_cost))
    print("Worst cost: " + str(worst_cost))
    print("Mean cost after 200 solutions: " + str(avg_cost))

    show_solution(path, best_sol, title="Best Tour")


def nn_solution(matrix, v1):
    nn_matrix = matrix.copy()
    n = math.ceil(matrix.shape[0] / 2)

    v = v1
    sol = [v1]

    # choosing the next n-1 solutions
    for i in range(n-1):
        nn = np.argmin(nn_matrix[v])
        nn_matrix[v, :] = np.inf
        nn_matrix[:, v] = np.inf
        v = nn
        sol.append(v)
    
    cost = sum(matrix[sol[i-1], sol[i]] for i in range(n))
    
    return sol, cost


def run_nn_exp(path: str):
    matrix = get_dist_matrix(path)

    solutions = []

    for v in range(200):
        solutions.append(nn_solution(matrix, v1=v))

    costs = np.array([cost for sol, cost in solutions])
    best_sol, best_cost = min(solutions, key=lambda x: x[1])
    worst_sol, worst_cost = max(solutions, key=lambda x: x[1])
    avg_cost = np.mean(costs)

    print("Best cost: " + str(best_cost))
    print("Worst cost: " + str(worst_cost))
    print("Mean cost after 200 solutions: " + str(avg_cost))

    show_solution(path, best_sol, title="Best Tour")
    show_solution(path, worst_sol, title="Worst Tour")


def greedy_solution(matrix_src, start_v):
    matrix = matrix_src.copy()
    n = math.ceil(matrix.shape[0] / 2)
    
    next_v = np.argmin(matrix[start_v])

    cycle = [start_v, next_v]
    cost = calculate_cost(cycle, matrix_src)
    matrix[:, [start_v, next_v]] = np.inf

    for _ in range(n - 2):
        best_new_cost = np.inf
        best_new_cycle = cycle
        chosen_v = 0

        for i, v in enumerate(cycle):
            closest_v = np.argmin(matrix[v])
            new_cycle = cycle[:i] + [closest_v] + cycle[i:]            
            new_cost = calculate_cost(new_cycle, matrix_src)

            if new_cost < best_new_cost:
                best_new_cycle = new_cycle
                best_new_cost = new_cost
                chosen_v = closest_v

        cycle = best_new_cycle
        cost = best_new_cost
        matrix[:, chosen_v] = np.inf

    return cycle, cost


def run_greedy_experiment(path: str):
    matrix = get_dist_matrix(path)

    solutions = []

    for v in range(200):
        solutions.append(greedy_solution(matrix, v))

    costs = np.array([cost for sol, cost in solutions])
    best_sol, best_cost = min(solutions, key=lambda x: x[1])
    worst_sol, worst_cost = max(solutions, key=lambda x: x[1])
    avg_cost = np.mean(costs)

    print("Best cost: " + str(best_cost))
    print("Worst cost: " + str(worst_cost))
    print("Mean cost after 200 solutions: " + str(avg_cost))

    show_solution(path, best_sol, title="Best Tour")
    show_solution(path, worst_sol, title="Worst Tour")
