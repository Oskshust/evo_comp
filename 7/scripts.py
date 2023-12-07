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


def destroy(solution):
    # 20-30% length of the solution
    subpath_length = np.random.randint(20, 31)
    
    start_index = np.random.randint(0, len(solution))
    
    for i in range(start_index, start_index + subpath_length):
        solution[i%len(solution)] = -1
    
    return solution


def find_regret_with_solution(solution, vertex_id, matrix):
    costs = []
    solutions = []
    for i in range(len(solution)+1):
        if vertex_id in solution:
            continue
        new_sol = solution[:i] + [vertex_id] + solution[i:]
        solutions.append(new_sol)
        costs.append(calculate_cost(new_sol, matrix))
    return get_regret_n_sol(costs, solutions)


def get_regret_n_sol(costs, solutions):
    first = np.argmin(costs)
    cost1 = costs[first]
    sol = solutions[first]
    costs = np.delete(costs, first)
    second = np.argmin(costs)
    cost2 = costs[second]
    return cost2 - cost1, sol


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
        new_costs = np.zeros(shape=unvisited.shape)
        new_sols = np.zeros(shape=unvisited.shape, dtype=np.ndarray)

        for vertex_id in np.where(unvisited == True)[0]:
            regret, solution = find_regret_with_solution(cycle, vertex_id, matrix)
            new_cost = calculate_cost(solution, matrix)
            increase = new_cost - current_cost
            
            score = 0.5 * regret - 0.5 * increase
            scores[vertex_id] = score
            new_sols[vertex_id] = solution
            new_costs[vertex_id] = new_cost

        highest_score_id = np.argmax(scores)
        cycle = new_sols[highest_score_id]
        unvisited[highest_score_id] = False
        current_cost = new_costs[highest_score_id]

    return cycle, current_cost


def lns(matrix, finish_time, with_ls):
    x, cost = steepest(matrix, random_solution(matrix))
        
    best_solution, best_cost = x.copy(), cost
    
    ils_iterations = 1
    while time.time() < finish_time:
        x = destroy(x)
        x, x_cost = repair(matrix, x)
        
        if with_ls:
            y, y_cost = steepest(matrix, x)
            if y_cost < best_cost:
                best_solution, best_cost = y.copy(), y_cost
                x = y
        else:
            if x_cost < best_cost:
                best_solution, best_cost = x.copy(), x_cost

        ils_iterations += 1

    return best_solution, best_cost, ils_iterations


# avg of 4 MSLS exps -> 7.2s/iteration -> 7.2s/i * 200i = 1440s   
def run_lns(path: str, n_runs=20, with_ls=True):
    max_time_per_run = 5 * 7.2
    matrix = get_dist_matrix(path)

    best_solutions = []
    avg_times = []
    avg_ils_iterations = []

    for _ in range(n_runs):
        start = time.time()

        best_solution, best_cost, ils_iterations = lns(matrix, start + max_time_per_run, with_ls)

        end = time.time()
        
        avg_time = (end - start) / ils_iterations
        avg_ils_iterations.append(ils_iterations)
        avg_times.append(avg_time)

        best_solutions.append((best_solution, best_cost))

    print(f"Average time per iteration: {np.mean(avg_times)} s")
    print(f"Average ILS iterations: {np.mean(avg_ils_iterations)}")
    summarize_results(best_solutions, path)
