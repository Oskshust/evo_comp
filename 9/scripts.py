import numpy as np
import time

from common import *


def destroy(solution):
    # 20-30% length of the solution
    subpath_length = np.random.randint(20, 31)
    
    start_index = np.random.randint(0, len(solution))
    
    for i in range(start_index, start_index + subpath_length):
        solution[i%len(solution)] = -1
    
    return solution


def find_regret_with_solution(solution, vertex_id, matrix):
    solutions = []
    deltas = []

    for i in range(len(solution)):
        if vertex_id in solution:
            continue

        new_sol = solution[:i] + [vertex_id] + solution[i:]
        solutions.append(new_sol)
        
        cost_out = matrix[solution[i - 1]][solution[i]]
        cost_in = matrix[solution[i - 1]][vertex_id] + matrix[vertex_id][solution[i]]
        
        deltas.append(cost_in - cost_out)

    return get_regret_n_sol(deltas, solutions)


def get_regret_n_sol(deltas, solutions):
    first = np.argmin(deltas)
    delta_1 = deltas[first]
    sol = solutions[first]
    deltas = np.delete(deltas, first)
    second = np.argmin(deltas)
    delta_2 = deltas[second]
    return delta_2 - delta_1, sol, delta_1


# weighted regret heuristic with weight =  0.5
def repair(matrix, destroyed_solution):
    cycle = [vertex for vertex in destroyed_solution if vertex != -1]
    
    current_cost = calculate_cost(cycle, matrix)

    unvisited = np.ones(len(matrix), dtype='bool')

    for id in range(len(destroyed_solution)):
        if destroyed_solution[id] != -1:
            unvisited[destroyed_solution[id]] = False

    while np.any(unvisited):
        scores = -np.ones(shape=unvisited.shape) * np.inf
        new_deltas = np.zeros(shape=unvisited.shape)
        new_sols = np.zeros(shape=unvisited.shape, dtype=np.ndarray)

        for vertex_id in np.where(unvisited == True)[0]:
            regret, solution, delta = find_regret_with_solution(cycle, vertex_id, matrix)

            score = 0.5 * regret - 0.5 * delta
            scores[vertex_id] = score
            new_sols[vertex_id] = solution
            new_deltas[vertex_id] = delta

        highest_score_id = np.argmax(scores)
        cycle = new_sols[highest_score_id]
        unvisited[highest_score_id] = False
        current_cost += new_deltas[highest_score_id]

    return cycle, current_cost


def operator1():
    return None


def operator2():
    return None


def get_child(parent_1, parent_2):
    # TODO
    return parent_1, parent_2


def breed(population, breed_ops=20):
    new_population = population.copy()
    for i in range(breed_ops):
        parent_1id, parent_2id = np.random.choice(len(population), 2, replace=False)
        parent_1, parent_2 = population[parent_1id], population[parent_2id]

        child_1, child_2 = get_child(parent_1, parent_2)

        new_population.append(child_1)
        new_population.append(child_2)

    new_population = sorted(new_population)

    return new_population


def hea(matrix, finish_time):
    population =  [random_solution(matrix) for i in range(20)]
        
    best_solution, best_cost = population[0][0].copy(), population[0][1]
    
    iterations = 1
    while time.time() < finish_time:
        
        population = breed(population)
        
        current_best, current_best_cost = population[0][0], population[0][1] 
        if current_best_cost < best_cost:
            best_solution, best_cost = current_best.copy(), current_best_cost

        iterations += 1

    return best_solution, best_cost, iterations


def run_hea(path: str, n_runs=20):
    max_time_per_run = 2 * 7.2
    matrix = get_dist_matrix(path)

    best_solutions = []
    avg_times = []
    avg_iterations = []

    for _ in range(n_runs):
        start = time.time()

        best_solution, best_cost, iterations = hea(matrix, start + max_time_per_run)

        end = time.time()
        
        avg_time = (end - start) / iterations
        avg_iterations.append(iterations)
        avg_times.append(avg_time)

        best_solutions.append((best_solution, best_cost))

    print(f"Average time per iteration: {np.mean(avg_times)} s")
    print(f"Average iterations: {np.mean(avg_iterations)}")
    summarize_results(best_solutions, path)
