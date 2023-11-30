import numpy as np
import matplotlib.pyplot as plt
import csv
import math


def apply_move(solution, move):
    if move[0] == "edge":
        new_solution = exchange_edges(solution, move[1], move[2])
    else:
        new_solution = insert_node(solution, move[1], move[2])

    return new_solution


def exchange_edges(solution, move_start, move_end):
    if move_start > move_end:
        neighbor = np.concatenate((solution[move_start:][::-1], solution[move_end+1:move_start], solution[:move_end+1][::-1]))
    else:
        neighbor = np.concatenate((solution[:move_start], solution[move_start:move_end+1][::-1], solution[move_end+1:]))
    
    return neighbor


def insert_node(solution, new_node, insert_id):
    neighbor = solution.copy()

    neighbor[insert_id] = new_node

    return neighbor


def calculate_delta_node(solution, matrix, new_node, insert_position):
    prev_vertex = solution[insert_position - 1]
    next_vertex = solution[(insert_position + 1) % len(solution)]

    removed_edges = ((prev_vertex, solution[insert_position]), (solution[insert_position], next_vertex))
    added_edges = ((prev_vertex, new_node), (new_node, next_vertex))

    cost_out = sum(matrix[e1][e2] for e1, e2 in removed_edges)
    cost_in = sum(matrix[e1][e2] for e1, e2 in added_edges)

    if math.isnan(cost_in - cost_out) or cost_in - cost_out == np.inf or cost_in - cost_out == -np.inf:
        return 0

    return cost_in - cost_out


def calculate_delta_edge(solution, matrix, start, end):
    prev_vertex = solution[start - 1]
    next_vertex = solution[(end + 1) % len(solution)]

    removed_edges = ((solution[start], prev_vertex), (solution[end], next_vertex))
    added_edges = ((solution[end], prev_vertex), (solution[start], next_vertex))

    cost_out = sum(matrix[e1][e2] for e1, e2 in removed_edges)
    cost_in = sum(matrix[e1][e2] for e1, e2 in added_edges)

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

    return sol
