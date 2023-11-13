import numpy as np
import time

from common import *


def steepest(matrix, starting_sol, use_previous_deltas):
    ...


def get_neighbourhood(solution, matrix):
    ...


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
