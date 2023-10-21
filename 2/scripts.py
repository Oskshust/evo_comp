import csv
import numpy as np
import math
import matplotlib.pyplot as plt


def get_coords_n_costs(path: str):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)

    data = np.array(data).astype(int)

    coords = data[:, :2]
    costs = data[:, 2]

    return coords, costs

def get_dist_matrix(path: str):
    coords, costs = get_coords_n_costs(path)

    distances = np.round(np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))).astype(float)
    distances[distances == 0] = np.inf

    return distances + costs

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

def find_regret_with_solution(solution, vertex_id, matrix):
    costs = []
    solutions = []
    
    # 1st idea
    # for i in range(len(solution)+1):
        # new_sol = solution[:i] + [vertex_id] + solution[i:]
        # solutions.append(new_sol)
        # costs.append(calculate_cost(new_sol, matrix))
    # return get_regret_n_sol(costs, solutions)
    
    # 2nd idea - just two closest possibilities.
    temp_matrix = matrix.copy()
    nn1_id = None
    while nn1_id not in solution:
        nn1_id = np.argmin(temp_matrix[vertex_id])
        temp_matrix[vertex_id][nn1_id] = np.inf
    nn2_id = None
    while nn2_id not in solution:
        nn2_id = np.argmin(temp_matrix[vertex_id])
        temp_matrix[vertex_id][nn2_id] = np.inf

    sol1 = solution[:nn1_id] + [vertex_id] + solution[nn1_id:]
    sol2 = solution[:nn2_id] + [vertex_id] + solution[nn2_id:]
    # solutions.append(sol1)
    # costs.append(calculate_cost(sol1, matrix))
    return calculate_cost(sol2, matrix) - calculate_cost(sol1, matrix), sol1
    

def get_regret_n_sol(costs, solutions):
    first = np.argmin(costs)
    cost1 = costs[first]
    sol = solutions[first]
    costs = np.delete(costs, first)
    solutions = np.delete(solutions, first)
    second = np.argmin(costs)
    cost2 = costs[second]
    return cost2 - cost1, sol

def regret2(matrix, start_v):
    n = math.ceil(matrix.shape[0] / 2)
    
    next_v = np.argmin(matrix[start_v])

    cycle = [start_v, next_v]

    unvisited = np.arange(len(matrix))
    unvisited = np.delete(unvisited, [start_v, next_v])

    for _ in range(n - 2):
        regrets = [np.inf]*len(unvisited)
        new_sols = [np.inf]*len(unvisited)
        
        for free_vertex_id in range(len(unvisited)):
            reg, sol = find_regret_with_solution(cycle, unvisited[free_vertex_id], matrix)
            regrets[free_vertex_id] = reg
            new_sols[free_vertex_id] = sol
        
        highest_reg_id = np.argmax(regrets)
        cycle = new_sols[highest_reg_id]
        unvisited = np.delete(unvisited, highest_reg_id)

    return cycle, calculate_cost(cycle, matrix)

def run_regret2_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(regret2(matrix, v))

    costs = np.array([cost for sol, cost in solutions])
    best_sol, best_cost = min(solutions, key=lambda x: x[1])
    worst_sol, worst_cost = max(solutions, key=lambda x: x[1])
    avg_cost = np.mean(costs)

    print("Best cost: " + str(best_cost))
    print("Worst cost: " + str(worst_cost))
    print("Mean cost after 200 solutions: " + str(avg_cost))

    show_solution(path, best_sol, title="Best Tour")
    show_solution(path, worst_sol, title="Worst Tour")