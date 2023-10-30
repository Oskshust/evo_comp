import csv
import numpy as np
import math
import matplotlib.pyplot as plt


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
    plt.figure(figsize=(10, 10))
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


def run_weighted_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(weighted_regret(matrix, v, 0.5))

    costs = np.array([cost for sol, cost in solutions])
    best_sol, best_cost = min(solutions, key=lambda x: x[1])
    worst_sol, worst_cost = max(solutions, key=lambda x: x[1])
    avg_cost = np.mean(costs)

    print("Best cost: " + str(best_cost))
    print("Worst cost: " + str(worst_cost))
    print("Mean cost after 200 solutions: " + str(avg_cost))

    show_solution(path, best_sol, title="Best Tour")
    show_solution(path, worst_sol, title="Worst Tour")


def random_solution(matrix):
    n = math.ceil(matrix.shape[0] / 2)

    sol = np.array(np.random.choice(matrix.shape[0], size=n, replace=False))

    cost = calculate_cost(sol, matrix)

    return sol, cost


def calculate_delta():
    return 0


def get_neighbourhood_2n(solution, num_neighbors=10):
    neighbors = []
    
    # intra-route
    for i in range(len(solution)):
        for j in range(i + 1, min(i + num_neighbors + 1, len(solution))):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
            if len(neighbors) >= num_neighbors:
                return neighbors
    
    all_nodes = set() 
    available_nodes = list(all_nodes - set(solution))
    
    # inter-route
    for i in range(len(solution)):
        for j in range(min(num_neighbors, len(available_nodes))):
            neighbor = solution.copy()
            neighbor[i] = available_nodes[j]
            neighbors.append(neighbor)
            if len(neighbors) >= num_neighbors:
                return neighbors
    
    return neighbors


def steepest_2n(matrix, starting_sol):
    best_sol = starting_sol
    neighbourhood = get_neighbourhood_2n(starting_sol)
    while len(neighbourhood):
        best_sol = neighbourhood[index]
        neighbourhood = get_neighbourhood_2n(best_sol)
        
    return None


def run_steepest_2n_r_experiment(path: str):
    matrix = get_dist_matrix(path)
    solutions = []

    for v in range(200):
        solutions.append(steepest_2n(matrix, random_solution(matrix)))

    costs = np.array([cost for sol, cost in solutions])
    best_sol, best_cost = min(solutions, key=lambda x: x[1])
    worst_sol, worst_cost = max(solutions, key=lambda x: x[1])
    avg_cost = np.mean(costs)

    print("Best cost: " + str(best_cost))
    print("Worst cost: " + str(worst_cost))
    print("Mean cost after 200 solutions: " + str(avg_cost))

    show_solution(path, best_sol, title="Best Tour")
    show_solution(path, worst_sol, title="Worst Tour")