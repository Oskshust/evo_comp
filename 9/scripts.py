import numpy as np
import time

from common import *


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


def random_repair(solution):
    child = solution.copy()
    for i in range(len(child)):
        available_nodes = list(set(range(200)) - set(child))
        if child[i] == -1:
            child[i] = np.random.choice(available_nodes, 1)[0]

    return child


def get_unfinished_child(parent_1, parent_2):
    child = np.array([-1 for i in range(100)])
    
    for i in range(len(parent_2) - 1):
        if parent_2[i] in parent_1 and parent_2[i+1] in parent_1:
            index_1 = np.where(parent_1 == parent_2[i])[0]
            index_2 = np.where(parent_1 == parent_2[i+1])[0]
            if abs(index_1 - index_2) == 1:
                child[index_1] = parent_1[index_1]
                child[index_2] = parent_1[index_2]

    for node in parent_2:
        if node in parent_1:
            index = np.where(parent_1 == node)[0][0]
            child[index] = node
    
    return child


def operator_random_rep(parent_1, parent_2, matrix):
    # We locate in the offspring all common nodes and edges and fill the rest of the solution at random
    child = random_repair(get_unfinished_child(parent_1, parent_2))

    return child, calculate_cost(child, matrix)


def operator_reg_rep(parent_1, parent_2, matrix):
    return repair(matrix, get_unfinished_child(parent_1, parent_2))


def get_children(parent_1, parent_2, matrix):
    child_1, cost_1 = operator_random_rep(parent_1, parent_2, matrix)
    child_2, cost_2 = operator_reg_rep(parent_1, parent_2, matrix)
    return child_1, cost_1, child_2, cost_2


def breed(population, matrix, population_size=10):
    new_population = population.copy()
    costs_of_population = set(cost for sol, cost in new_population)
    i = 0
    while i < population_size:
        parent_1id, parent_2id = np.random.choice(len(population), 2, replace=False)
        parent_1, parent_2 = population[parent_1id][0], population[parent_2id][0]

        child_1, cost_1, child_2, cost_2 = get_children(parent_1, parent_2, matrix)
        while cost_1 in costs_of_population:
            child_1, cost_1 = operator_random_rep(parent_1, parent_2, matrix)
        while cost_2 in costs_of_population:
            child_2, cost_2 = operator_reg_rep(parent_2, parent_1, matrix)
        i += 1

        new_population.append((child_1, cost_1))
        new_population.append((child_2, cost_2))

    new_population = sorted(new_population, key=lambda x: x[1])

    return new_population[:population_size]


def hea(matrix, finish_time):
    population = [steepest(matrix, random_solution(matrix)[0]) for i in range(20)]
    best_solution, best_cost = population[0][0].copy(), population[0][1]
    
    iterations = 1
    while time.time() < finish_time:
        
        population = breed(population, matrix)
        
        current_best, current_best_cost = population[0][0], population[0][1] 
        if current_best_cost < best_cost:
            best_solution, best_cost = current_best.copy(), current_best_cost

        iterations += 1

    return best_solution, best_cost, iterations


def run_hea(path: str, n_runs=20):
    max_time_per_run = 200 * 7.2
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
