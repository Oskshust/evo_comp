import csv
import numpy as np
import math
import matplotlib.pyplot as plt


def summarize_results(solutions, path):
    costs = np.array([cost for sol, cost in solutions])
    best_sol, best_cost = min(solutions, key=lambda x: x[1])
    worst_sol, worst_cost = max(solutions, key=lambda x: x[1])
    avg_cost = np.mean(costs)

    print("Best cost: " + str(best_cost))
    print("Worst cost: " + str(worst_cost))
    print("Mean cost after 200 solutions: " + str(avg_cost))

    show_solution(path, best_sol, title="Best Tour")


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


def show_solution(path, solution, title):
    coords, node_costs = get_coords_n_costs(path)
    # plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c=node_costs, cmap='plasma')

    best_tour_coords = np.append(solution, solution[0])
    plt.plot(coords[best_tour_coords, 0], coords[best_tour_coords, 1], 'r-')

    plt.colorbar(label='Cost')
    plt.title(title)
    plt.show()


def find_regret_with_solution(solution, vertex_id, matrix):
    costs = []
    solutions = []
    for i in range(len(solution)+1):
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


def weighted_regret(matrix, start_v, weight):
    n = math.ceil(matrix.shape[0] / 2)
    
    next_v = np.argmin(matrix[start_v])

    cycle = [start_v, next_v]
    current_cost = calculate_cost(cycle, matrix)

    unvisited = np.ones(len(matrix), dtype='bool')
    unvisited[start_v] = False
    unvisited[next_v] = False

    for _ in range(n - 2):
        scores = -np.ones(shape=unvisited.shape) * np.inf
        new_costs = np.zeros(shape=unvisited.shape)
        new_sols = np.zeros(shape=unvisited.shape, dtype=np.ndarray)

        for vertex_id in np.where(unvisited == True)[0]:
            regret, solution = find_regret_with_solution(cycle, vertex_id, matrix)
            new_cost = calculate_cost(solution, matrix)
            increase = new_cost - current_cost
            
            score = weight * regret - (1 - weight) * increase
            scores[vertex_id] = score
            new_sols[vertex_id] = solution
            new_costs[vertex_id] = new_cost

        highest_score_id = np.argmax(scores)
        cycle = new_sols[highest_score_id]
        unvisited[highest_score_id] = False
        current_cost = new_costs[highest_score_id]

    return cycle, current_cost


def random_solution(matrix):
    n = math.ceil(matrix.shape[0] / 2)

    sol = np.array(np.random.choice(matrix.shape[0], size=n, replace=False))

    cost = calculate_cost(sol, matrix)

    return sol, cost


# m_id - id of vertex martix-wise, s_id - id of vertex solution-wise,
def calculate_delta(solution, matrix, m_id_in, s_id_out):
    prev_vertex = solution[s_id_out - 1]
    next_vertex = solution[(s_id_out + 1) % len(solution)]

    cost_out = matrix[prev_vertex][solution[s_id_out]] + matrix[solution[s_id_out]][next_vertex]

    cost_in = matrix[prev_vertex][m_id_in] + matrix[m_id_in][next_vertex]
    
    if math.isnan(cost_in - cost_out) or cost_in-cost_out==np.inf:
        return 0

    return cost_in - cost_out


def calculate_delta_edge(solution, matrix, start, end):
    cost_out = matrix[solution[start - 1]][solution[start]] + matrix[solution[end]][solution[(end + 1) % len(solution)]]
    cost_in = matrix[solution[start - 1]][solution[end]] + matrix[solution[start]][solution[(end + 1) % len(solution)]]

    if math.isnan(cost_in - cost_out) or cost_in-cost_out==np.inf:
        return 0

    return cost_in - cost_out


def get_neighbourhood_2n(solution, matrix):
    neighbors = []
    solution_length = len(solution)
    matrix_shape = matrix.shape[0]

    # intra-route
    for i in range(solution_length-1):
        for j in range(i + 1, solution_length):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            delta = calculate_delta(solution, matrix, neighbor[i], j) + calculate_delta(solution, matrix, neighbor[j], i)
            neighbors.append((neighbor, delta))
 
    all_nodes = set(range(matrix_shape))
    available_nodes = all_nodes - set(solution)

    # inter-route
    for i in range(solution_length):
        for node in available_nodes:
            neighbor = solution.copy()
            neighbor[i] = node
            delta = calculate_delta(solution, matrix, node, i)
            neighbors.append((neighbor, delta))

    return neighbors


def get_neighbourhood_2e(solution, matrix):
    neighbors = []
    solution_length = len(solution)
    matrix_shape = matrix.shape[0]

    # intra-route
    for i in range(solution_length-1):
        for j in range(i + 1, solution_length):
            neighbor = np.concatenate((solution[:i], solution[i:j+1][::-1], solution[j+1:]))

            delta = calculate_delta_edge(solution, matrix, i, j)
            neighbors.append((neighbor, delta))
 
    all_nodes = set(range(matrix_shape))
    available_nodes = all_nodes - set(solution)

    # inter-route
    for i in range(solution_length):
        for node in available_nodes:
            neighbor = solution.copy()
            neighbor[i] = node
            delta = calculate_delta(solution, matrix, node, i)
            neighbors.append((neighbor, delta))

    return neighbors


def steepest_2n(matrix, starting_sol):
    best_sol = starting_sol
    best_delta = 0
    neighbourhood = get_neighbourhood_2n(starting_sol, matrix)

    while len(neighbourhood):
        deltas = np.array([delta for _, delta in neighbourhood])
        best_index = np.argmin(deltas)
        best_sol, best_delta = neighbourhood[best_index]

        if best_delta >= 0:
            break

        neighbourhood = get_neighbourhood_2n(best_sol, matrix)

    return best_sol, calculate_cost(best_sol, matrix)


def steepest_2e(matrix, starting_sol):
    best_sol = starting_sol
    best_delta = 0
    neighbourhood = get_neighbourhood_2e(starting_sol, matrix)
    
    while len(neighbourhood):
        deltas = np.array([delta for _, delta in neighbourhood])
        best_index = np.argmin(deltas)
        best_sol, best_delta = neighbourhood[best_index]

        if best_delta >= 0:
            break

        neighbourhood = get_neighbourhood_2e(best_sol, matrix)

    return best_sol, calculate_cost(best_sol, matrix)


def get_random_neighbour_2n(solution, matrix):
    neighbor = solution.copy()
    all_nodes = set(range(matrix.shape[0]))
    available_nodes = list(all_nodes - set(solution))

    action = np.random.choice(['swap', 'replace'])

    if action == 'swap':
        i, j = np.random.choice(len(solution), size=2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        delta = calculate_delta(solution, matrix, neighbor[i], j) + calculate_delta(solution, matrix, neighbor[j], i)
    else:
        i = np.random.choice(len(solution))
        j = np.random.choice(len(available_nodes))
        neighbor[i] = available_nodes[j]
        delta = calculate_delta(solution, matrix, available_nodes[j], i)

    return neighbor, delta


def get_random_neighbour_2e(solution, matrix):
    neighbor = solution.copy()
    all_nodes = set(range(matrix.shape[0]))
    available_nodes = list(all_nodes - set(solution))

    action = np.random.choice(['swap', 'replace'])

    if action == 'swap':
        i, j = np.random.choice(len(solution), size=2, replace=False)

        neighbor = np.concatenate((solution[:i], solution[i:j+1][::-1], solution[j+1:]))
        delta = calculate_delta_edge(solution, matrix, i, j)

    else:
        i = np.random.choice(len(solution))
        j = np.random.choice(len(available_nodes))
        neighbor[i] = available_nodes[j]
        delta = calculate_delta(solution, matrix, available_nodes[j], i)

    return neighbor, delta


def greedy_2n(matrix, starting_sol):
   best_sol = starting_sol
   best_delta = 0
   probably_best_sol, best_delta = get_random_neighbour_2n(starting_sol, matrix)
   
   while best_delta < 0:
       best_sol, best_delta = probably_best_sol, best_delta
       probably_best_sol, best_delta = get_random_neighbour_2n(best_sol, matrix)
       
   return best_sol, calculate_cost(best_sol, matrix)


def greedy_2e(matrix, starting_sol):
   best_sol = starting_sol
   best_delta = 0
   probably_best_sol, best_delta = get_random_neighbour_2e(starting_sol, matrix)
   
   while best_delta < 0:
       best_sol, best_delta = probably_best_sol, best_delta
       probably_best_sol, best_delta = get_random_neighbour_2e(best_sol, matrix)
       
   return best_sol, calculate_cost(best_sol, matrix)


def run_greedy_2n_r_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(greedy_2n(matrix, random_solution(matrix)[0]))

    summarize_results(solutions, path)


def run_greedy_2n_bgch_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(greedy_2n(matrix, weighted_regret(matrix, v, 0.5)[0]))

    summarize_results(solutions, path)


def run_steepest_2n_r_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(2):
        solutions.append(steepest_2n(matrix, random_solution(matrix)[0]))

    summarize_results(solutions, path)


def run_steepest_2n_bgch_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(steepest_2n(matrix, weighted_regret(matrix, v, 0.5)[0]))

    summarize_results(solutions, path)


def run_greedy_2e_r_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(greedy_2e(matrix, random_solution(matrix)[0]))

    summarize_results(solutions, path)


def run_greedy_2e_bgch_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(greedy_2e(matrix, weighted_regret(matrix, v, 0.5)[0]))

    summarize_results(solutions, path)


def run_steepest_2e_r_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(2):
        solutions.append(steepest_2e(matrix, random_solution(matrix)[0]))

    summarize_results(solutions, path)


def run_steepest_2e_bgch_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(steepest_2e(matrix, weighted_regret(matrix, v, 0.5)[0]))

    summarize_results(solutions, path)
