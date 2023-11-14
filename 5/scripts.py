import numpy as np
import time

from common import *


def get_neighborhood(solution, matrix, only_improving):
    solution_length = len(solution)
    neighbors = []

    # intra
    for i in range(solution_length - 1):
        for j in range(i + 1, solution_length):
            neighbor, delta = exchange_edges(solution, matrix, i, j)

            if not only_improving or delta < 0:
                neighbors.append((neighbor, delta))

    all_nodes = set(range(len(matrix)))
    available_nodes = all_nodes - set(solution)

    # inter
    for i in range(solution_length):
        for node in available_nodes:
            neighbor, delta = insert_node(solution, matrix, node, i)

            if not only_improving or delta < 0:
                neighbors.append((neighbor, delta))

    return neighbors


def is_move_applicable():
    ...


def steepest(matrix, starting_sol, use_previous_deltas):
    best_sol = np.array(starting_sol)
    best_delta = 0
    neighbourhood = get_neighborhood(best_sol, matrix, use_previous_deltas)

    while len(neighbourhood):
        deltas = np.array([delta for _, delta in neighbourhood])
        best_index = np.argmin(deltas)
        probably_best_sol, best_delta = neighbourhood[best_index]

        if best_delta >= 0:
            break

        best_sol, best_delta = neighbourhood[best_index]
        neighbourhood = get_neighborhood(best_sol, matrix, use_previous_deltas)

    return best_sol, calculate_cost(best_sol, matrix)


def run_experiment(path: str, n_iterations=200, use_previous_deltas=False):
    matrix = get_dist_matrix(path)
    solutions = []

    start = time.time()

    for _ in range(n_iterations):
        starting_solution = random_solution(matrix)[0]
        solution = steepest(matrix, starting_solution, use_previous_deltas)
        solutions.append(solution)

    end = time.time()

    print(f"Time per iteration: {(end - start)/n_iterations} s")
    summarize_results(solutions, path)


def run_baseline(path: str, n_iterations=200):
    run_experiment(path, n_iterations)
