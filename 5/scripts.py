import numpy as np
import time

from common import *


def get_neighborhood(solution, matrix, only_improving, previous_neighbors = None):
    solution_length = len(solution)

    neighbors = []
    if previous_neighbors != None:
        for move, _ in previous_neighbors:
            new_delta = update_delta(solution, matrix, move)
            if new_delta < 0:
                neighbors.append((move, new_delta))

    old_moves = {move for move, _ in neighbors}

    # intra
    for i in range(solution_length - 1):
        for j in range(i + 1, solution_length):
            delta, removed_edges = calculate_delta_edge(solution, matrix, i, j)

            if not only_improving or delta < 0:
                move = ("edge", removed_edges, i, j)
                
                if move not in old_moves:
                    neighbors.append((move, delta))

    all_nodes = set(range(len(matrix)))
    available_nodes = all_nodes - set(solution)

    # inter
    for i in range(solution_length):
        for node in available_nodes:
            delta, removed_edges = calculate_delta_node(solution, matrix, node, i)

            if not only_improving or delta < 0:
                move = ("node", removed_edges, node, i)
                
                if move not in old_moves:
                    neighbors.append((move, delta))

    if only_improving:
        neighbors = sorted(neighbors, key=lambda x: x[1])

    return neighbors


def get_neighborhood_new(solution, matrix, previous_neighbors, moves_to_recalc):
    solution_length = len(solution)

    neighbors = [nb for nb in previous_neighbors]
    # if previous_neighbors != None:
    #     for move, _ in previous_neighbors:
    #         new_delta = update_delta(solution, matrix, move)
    #         if new_delta < 0:
    #             neighbors.append((move, new_delta))

    old_moves = {move for move, _ in neighbors}

    # intra
    for rec in moves_to_recalc:
        if rec[0] == "edge":
            for j in range(rec[1] + 1, solution_length):
                delta, removed_edges = calculate_delta_edge(solution, matrix, rec[1], j)

                if delta < 0:
                    move = ("edge", removed_edges, rec[1], j)
                    
                    if move not in old_moves:
                        neighbors.append((move, delta))

        else:
            all_nodes = set(range(len(matrix)))
            available_nodes = all_nodes - set(solution)

            # inter
            for node in available_nodes:
                delta, removed_edges = calculate_delta_node(solution, matrix, node, rec[1])

                if delta < 0:
                    move = ("node", removed_edges, node, rec[1])
                    
                    if move not in old_moves:
                        neighbors.append((move, delta))

    return sorted(neighbors, key=lambda x: x[1])


def update_delta(new_solution, matrix, old_move):
    if old_move[0] == "edge":
        new_delta, _ = calculate_delta_edge(new_solution, matrix, old_move[2], old_move[3])
    else:
        new_delta, _ = calculate_delta_node(new_solution, matrix, old_move[2], old_move[3])

    return new_delta


def steepest(matrix, use_previous_deltas):
    starting_sol = random_solution(matrix)[0]
    best_sol = np.array(starting_sol)
    best_delta = 0

    neighbourhood = get_neighborhood(best_sol, matrix, use_previous_deltas)

    if use_previous_deltas:
        while len(neighbourhood):
            leftover_moves = []
            solution_edges = {(e1, e2) for e1, e2 in zip(best_sol, best_sol[1:])}
            solution_edges.add((best_sol[-1], best_sol[0]))
        
            solution_edges_reversed = {(e2, e1) for e1, e2 in solution_edges}
            
            best_move_found = False
            best_delta = None

            moves_to_recalc = set()

            for move, delta in neighbourhood:
                removed_edges = move[1]

                if best_move_found:
                    leftover_moves.append((move, delta))
                    continue

                should_move_be_removed = False

                if move[0] == "node" and move[2] in best_sol:
                    moves_to_recalc.add(("node", move[3]))
                    should_move_be_removed = True
                elif all(e in solution_edges for e in removed_edges) or all(e in solution_edges_reversed for e in removed_edges):
                    best_move = move
                    best_move_found = True
                    should_move_be_removed = True
                    moves_to_recalc.add(("edge", move[2]))
                    moves_to_recalc.add(("edge", move[3]))
                    # print(f"found best move, delta {delta}")

                if not should_move_be_removed:
                    leftover_moves.append((move, delta))

            if not best_move_found:
                break

            if best_move[0] == "edge":
                best_sol = exchange_edges(best_sol, best_move[2], best_move[3])
                # assert len(set(best_sol)) == len(best_sol), "edge"
            else:
                best_sol = insert_node(best_sol, best_move[2], best_move[3])
                # assert len(set(best_sol)) == len(best_sol), "node"
            neighbourhood = get_neighborhood_new(best_sol, matrix, leftover_moves, moves_to_recalc)
            # print(f"new best sol - {calculate_cost(best_sol, matrix)}")
    else:
        while len(neighbourhood):
            deltas = np.array([delta for _, delta in neighbourhood])
            best_index = np.argmin(deltas)
            _, best_delta = neighbourhood[best_index]

            if best_delta >= 0:
                break

            best_move, best_delta = neighbourhood[best_index]

            if best_move[0] == "edge":
                best_sol = exchange_edges(best_sol, best_move[2], best_move[3])
            else:
                best_sol = insert_node(best_sol, best_move[2], best_move[3])

            neighbourhood = get_neighborhood(best_sol, matrix, use_previous_deltas)
    
    return best_sol, calculate_cost(best_sol, matrix)


def run_experiment(path: str, n_iterations=200, use_previous_deltas=False):
    matrix = get_dist_matrix(path)
    solutions = []

    start = time.time()

    for _ in range(n_iterations):
        solution = steepest(matrix, use_previous_deltas)
        solutions.append(solution)

    end = time.time()

    print(f"Time per iteration: {(end - start)/n_iterations} s")
    summarize_results(solutions, path)
