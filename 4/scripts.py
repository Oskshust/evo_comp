import numpy as np

from common import *


def steepest(matrix, starting_sol, candidates):
    best_sol = np.array(starting_sol)
    best_delta = 0
    neighbourhood = get_neighbourhood(best_sol, matrix, candidates)

    while len(neighbourhood):
        deltas = np.array([delta for _, delta in neighbourhood])
        best_index = np.argmin(deltas)
        probably_best_sol, best_delta = neighbourhood[best_index]

        if best_delta >= 0:
            break

        best_sol, best_delta = neighbourhood[best_index]
        neighbourhood = get_neighbourhood(best_sol, matrix, candidates)

    return best_sol, calculate_cost(best_sol, matrix)


def get_neighbourhood(solution, matrix, candidates):
    if candidates is not None:
        neighbors = get_candidate_neighborhood(solution, matrix, candidates)
    else:
        neighbors = get_whole_neighborhood(solution, matrix)

    return neighbors


def get_whole_neighborhood(solution, matrix):
    solution_length = len(solution)
    neighbors = []

    # intra
    for i in range(solution_length-1):
        for j in range(i + 1, solution_length):
            neighbors.append(exchange_edges(solution, matrix, i, j))

    all_nodes = set(range(len(matrix)))
    available_nodes = all_nodes - set(solution)

    # inter
    for i in range(solution_length):
        for node in available_nodes:            
            neighbors.append(insert_node(solution, matrix, node, i))

    return neighbors


def exchange_edges(solution, matrix, move_start, move_end):
    neighbor = np.concatenate((solution[:move_start], solution[move_start:move_end+1][::-1], solution[move_end+1:]))
    delta = calculate_delta_edge(solution, matrix, move_start, move_end)

    return neighbor, delta


def insert_node(solution, matrix, new_node, insert_id):
    neighbor = solution.copy()
    neighbor[insert_id] = new_node
    delta = calculate_delta(solution, matrix, new_node, insert_id)

    return neighbor, delta


def get_candidate_neighborhood(solution, matrix, candidates):
    neighbors = []

    for n1_i, n1 in enumerate(solution):
        node_candidates = candidates[n1]
        next_i = (n1_i + 1) % len(solution)

        for n2 in node_candidates:
            where_n2 = np.where(solution == n2)[0]
            n2_in_solution = len(where_n2) != 0

            if n2_in_solution:
                n2_i = where_n2[0]

                neighbors.append(exchange_edges(solution, matrix, next_i, n2_i))
                neighbors.append(exchange_edges(solution, matrix, n1_i, n2_i - 1))
            else:
                neighbors.append(insert_node(solution, matrix, n2, next_i))
    
    return neighbors


def calculate_n_candidates(matrix, n):
    if n == 0:
        return None

    candidates = matrix.argsort(-1)[:,:n]
    return candidates


def run_experiment(path: str, n_iterations=200, n_candidates=10):
    matrix = get_dist_matrix(path)
    solutions = []

    candidates = calculate_n_candidates(matrix, n_candidates)

    for _ in range(n_iterations):
        starting_solution = random_solution(matrix)[0]
        solution = steepest(matrix, starting_solution, candidates)
        solutions.append(solution)

    summarize_results(solutions, path)


def run_baseline(path: str, n_iterations=200):
    run_experiment(path, n_iterations, 0)
