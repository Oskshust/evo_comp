import numpy as np
import matplotlib.pyplot as plt
import csv
import math


def calculate_delta_node(solution, matrix, new_node, insert_position, node_swap):
    prev_vertex = solution[insert_position - 1]
    next_vertex = solution[(insert_position + 1) % len(solution)]

    cost_out = matrix[prev_vertex][solution[insert_position]] + matrix[solution[insert_position]][next_vertex]
    
    if not node_swap:
        next_next_vertex = solution[(insert_position + 2) % len(solution)]
        cost_out += matrix[next_vertex][next_next_vertex]
        cost_in = matrix[prev_vertex][next_vertex] + matrix[next_vertex][new_node] + matrix[new_node][next_next_vertex]
    else:
        cost_in = matrix[prev_vertex][new_node] + matrix[new_node][next_vertex]

    if math.isnan(cost_in - cost_out) or cost_in-cost_out==np.inf:
        return 0

    return cost_in - cost_out


def calculate_delta_edge(solution, matrix, start, end):
    prev_vertex = solution[start - 1]
    next_vertex = solution[(end + 1) % len(solution)]

    cost_out = matrix[solution[start]][prev_vertex] + matrix[solution[end]][next_vertex]
    cost_in = matrix[solution[end]][prev_vertex] + matrix[solution[start]][next_vertex]

    if math.isnan(cost_in - cost_out) or cost_in-cost_out==np.inf:
        return 0

    return cost_in - cost_out


def summarize_results(solutions, path):
    costs = np.array([cost for sol, cost in solutions])
    best_sol, best_cost = min(solutions, key=lambda x: x[1])
    worst_sol, worst_cost = max(solutions, key=lambda x: x[1])
    avg_cost = np.mean(costs)

    print(f"Results: {round(avg_cost, 3)}({int(best_cost)} - {int(worst_cost)})")
    print(f"Best solution: {sorted(list(best_sol))}")

    show_solution(path, best_sol, title="Best Tour")


def show_solution(path, solution, title):
    coords, node_costs = get_coords_n_costs(path)
    # plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c=node_costs, cmap='plasma')

    best_tour_coords = np.append(solution, solution[0])
    plt.plot(coords[best_tour_coords, 0], coords[best_tour_coords, 1], 'r-')

    plt.colorbar(label='Cost')
    plt.title(title)
    plt.show()


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


def random_solution(matrix):
    n = math.ceil(matrix.shape[0] / 2)

    sol = np.array(np.random.choice(matrix.shape[0], size=n, replace=False))

    cost = calculate_cost(sol, matrix)

    return sol, cost
