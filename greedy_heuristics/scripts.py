import csv
import numpy as np
import random
import matplotlib.pyplot as plt


def get_dist_matrix(path: str):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)

    data = np.array(data).astype(int)

    coords = data[:, :2]
    costs = data[:, 2]

    distances = np.round(np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))).astype(int)

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
    n =  matrix.shape[0] // 2

    # I wonder whether it should be even simpler or it is ok to leave it in this form
    sol = np.array(random.sample(range(matrix.shape[0]), n))
    
    cost = sum(matrix[sol[i-1], sol[i]] for i in range(n))
    
    return sol, cost


def show_solution(path, solution):
    coords, node_costs = get_coords_n_costs(path)
    plt.scatter(coords[:, 0], coords[:, 1], c=node_costs, cmap='gray')

    best_tour_coords = np.append(solution, solution[0])
    plt.plot(coords[best_tour_coords, 0], coords[best_tour_coords, 1], 'r-')

    plt.colorbar(label='Cost')
    plt.title('Best Tour')
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
    
    show_solution(path, best_sol)