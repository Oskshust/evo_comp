import numpy as np
import time

from common import *


def get_neighborhood(solution, matrix):
    solution_length = len(solution)

    all_nodes = set(range(len(matrix)))
    available_nodes = all_nodes - set(solution)

    neighbors = [(("edge", i, j), calculate_delta_edge(solution, matrix, i, j)) for i in range(solution_length - 1) for j in range(i + 1, solution_length)]    
    neighbors = neighbors + [(("node", node, i), calculate_delta_node(solution, matrix, node, i)) for i in range(solution_length) for node in available_nodes]

    return neighbors


def steepest(matrix):
    starting_sol = random_solution(matrix)
    best_sol = np.array(starting_sol)
    best_delta = 0

    neighbourhood = get_neighborhood(best_sol, matrix)

    while len(neighbourhood):
        deltas = np.array([delta for _, delta in neighbourhood])
        best_index = np.argmin(deltas)
        _, best_delta = neighbourhood[best_index]

        if best_delta >= 0:
            break

        best_move, best_delta = neighbourhood[best_index]

        best_sol = apply_move(best_sol, best_move)

        neighbourhood = get_neighborhood(best_sol, matrix)

    return best_sol, calculate_cost(best_sol, matrix)


def run_experiment(path: str, n_iterations=200, n_runs=20):
    matrix = get_dist_matrix(path)

    best_solutions = []
    avg_times = []

    for _ in range(n_runs):
        solutions = []
        start = time.time()

        for _ in range(n_iterations):
            solution = steepest(matrix)
            solutions.append(solution)

        end = time.time()
        
        avg_time = (end - start) / n_iterations
        avg_times.append(avg_time)

        best_solution = min(solutions, key=lambda x: x[1])
        best_solutions.append(best_solution)

    print(f"Average time per iteration: {np.mean(avg_times)} s")
    summarize_results(best_solutions, path)
