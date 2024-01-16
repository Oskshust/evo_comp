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


def steepest(matrix, starting_sol):
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


def multi_segment_shuffle_perturbation(solution):
    new_solution = solution.copy()
    num_segments = np.random.randint(2, len(solution)//2)
    
    for _ in range(num_segments):
        node_indices = np.sort(np.random.choice(len(solution), 2, replace=False))
        np.random.shuffle(new_solution[node_indices[0]:node_indices[1]])

    return new_solution


def algo(matrix, start_time, time_limit):    
    y, cost = steepest(matrix, random_solution(matrix))
    best_solution, best_cost = y, cost
    
    ils_iterations = 1
    finish_time = time_limit + start_time
    while time.time() < finish_time:
        x = multi_segment_shuffle_perturbation(best_solution)
        y, y_cost = steepest(matrix, x)
        if y_cost < best_cost:
            best_solution, best_cost = y, y_cost
        ils_iterations += 1

    return best_solution, best_cost, ils_iterations

# def algo(matrix, start_time, time_limit):    
#     y, cost = steepest(matrix, random_solution(matrix))
#     best_solution, best_cost = y, cost
    
#     ils_iterations = 1
#     num_shuffles = 2
#     finish_time = time_limit + start_time
#     while time.time() < finish_time:
#         x = double_bridge_move(best_solution)
#         y, y_cost = steepest(matrix, x)
#         if y_cost < best_cost:
#             best_solution, best_cost = y, y_cost
#             num_shuffles = 2  # reset the number of shuffles
#         else:
#             num_shuffles += 1  # increase the number of shuffles
#         ils_iterations += 1

#     return best_solution, best_cost, ils_iterations


# avg of 4 MSLS exps -> 7.2s/iteration -> 7.2s/i * 200i = 1440s   
def run_algo(path: str, n_runs=20):
    max_time_per_run = 200 * 7.2
    matrix = get_dist_matrix(path)

    best_solutions = []
    avg_times = []
    avg_ils_iterations = []

    for _ in range(n_runs):
        start = time.time()

        best_solution, best_cost, ils_iterations = algo(matrix, start, max_time_per_run)

        end = time.time()
        
        avg_time = (end - start) / ils_iterations
        avg_ils_iterations.append(ils_iterations)
        avg_times.append(avg_time)

        best_solutions.append((best_solution, best_cost))

    print(f"Average time per iteration: {np.mean(avg_times)} s")
    print(f"Average ILS iterations: {np.mean(avg_ils_iterations)}")
    summarize_results(best_solutions, path)
