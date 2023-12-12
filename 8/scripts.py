import numpy as np
import matplotlib.pyplot as plt
from common import *


def calculate_similarity(solution_a, solution_b, measure="nodes"):
    if measure == "nodes":
        similarity = len(set(solution_a).intersection(set(solution_b)))
    else:
        edges_a = set((solution_a[i - 1], solution_a[i]) for i in range(len(solution_a)))
        edges_b = set((solution_b[i - 1], solution_b[i]) for i in range(len(solution_b)))
        similarity = len(edges_a.intersection(edges_b))
    return similarity


def get_neighborhood(solution, matrix):
    solution_length = len(solution)

    all_nodes = set(range(len(matrix)))
    available_nodes = all_nodes - set(solution)

    neighbors = [(("edge", i, j), calculate_delta_edge(solution, matrix, i, j)) for i in range(solution_length - 1) for j in range(i + 1, solution_length)]    
    neighbors = neighbors + [(("node", node, i), calculate_delta_node(solution, matrix, node, i)) for i in range(solution_length) for node in available_nodes]

    return neighbors


def greedy(matrix, starting_sol):
    best_sol = np.array(starting_sol)

    neighbourhood = get_neighborhood(best_sol, matrix)

    while len(neighbourhood):
        deltas = np.array([delta for _, delta in neighbourhood])
        negative_deltas = np.where(deltas < 0)[0]
        np.random.shuffle(negative_deltas) # so that we might explore more diverse moves

        if len(negative_deltas) == 0:
            break

        greedy_index = negative_deltas[0]

        greedy_move, _ = neighbourhood[greedy_index]

        best_sol = apply_move(best_sol, greedy_move)

        neighbourhood = get_neighborhood(best_sol, matrix)

    return best_sol, calculate_cost(best_sol, matrix)


def run_global_convexity(path: str, n_runs=1000):
    matrix = get_dist_matrix(path)

    local_optima = []
    local__optima_costs = []
    best_solution = None
    best_cost = np.inf

    for _ in range(n_runs):
        starting_solution = random_solution(matrix)

        solution, cost = greedy(matrix, starting_solution)

        if cost < best_cost:
            best_solution = solution
            best_cost = cost

        local_optima.append(solution)
        local__optima_costs.append(cost)

    similarities = {
        "best": {
            "nodes": [],
            "edges": []
        },
        "average": {
            "nodes": [],
            "edges": []
        }
    }

    for solution_a in local_optima:
        for measure in "nodes", "edges":
            similarities["best"][measure].append(calculate_similarity(solution_a, best_solution, measure))
            similarities["average"][measure].append(sum([calculate_similarity(solution_a, b, measure) for b in local_optima]) / n_runs)

    return similarities, local__optima_costs


def draw_charts(path: str, n_runs=1000):
    sims, f_values = run_global_convexity(path, n_runs)
    nodes_avg = sims['average']['nodes']
    nodes_best = sims['best']['nodes']
    edges_avg = sims['average']['edges']
    edges_best = sims['best']['edges']

    fig, axs = plt.subplots(4, figsize=(10, 10))

    sorted_pairs = sorted(zip(f_values, nodes_avg))
    sorted_f_values, sorted_data = zip(*sorted_pairs)

    axs[0].scatter(sorted_data, sorted_f_values, color='g')
    axs[0].set_title('Nodes')
    axs[0].set_xlabel('Objective Function Value')
    axs[0].set_ylabel('Avg Similarity')

    sorted_pairs = sorted(zip(f_values, edges_avg))
    sorted_f_values, sorted_data = zip(*sorted_pairs)

    axs[1].scatter(sorted_data, sorted_f_values, color='b')
    axs[1].set_title('Edges')
    axs[1].set_xlabel('Objective Function Value')
    axs[1].set_ylabel('Avg Similarity')

    sorted_pairs = sorted(zip(f_values, nodes_best))
    sorted_f_values, sorted_data = zip(*sorted_pairs)

    axs[2].scatter(sorted_data, sorted_f_values, color='g')
    axs[2].set_title('Nodes')
    axs[2].set_xlabel('Objective Function Value')
    axs[2].set_ylabel('Best Similarity')

    sorted_pairs = sorted(zip(f_values, edges_best))
    sorted_f_values, sorted_data = zip(*sorted_pairs)

    axs[3].scatter(sorted_data, sorted_f_values, color='b')
    axs[3].set_title('Edges')
    axs[3].set_xlabel('Objective Function Value')
    axs[3].set_ylabel('Best Similarity')

    plt.tight_layout()
    plt.show()
