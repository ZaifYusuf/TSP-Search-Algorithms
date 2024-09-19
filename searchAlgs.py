import argparse
import os
import random
import heapq
import networkx as nx
import numpy as np

def turn_txt_into_2d_matrix(filename):
    with open(filename, 'r') as file:
        n = int(file.readline().strip())
        
        matrix = []  
        
        for _ in range(n): 
            line = file.readline().strip()
            row = list(map(int, line.split()))
            matrix.append(row)
    
    return matrix


def NN(matrix):
    start = 0
    visited = [0]
    unvisited = list(range(1, len(matrix)))
    curr_node = start
    cost = 0
    while unvisited:
        min = float('inf')
        min_node = None
        for i in unvisited:
            i_min = matrix[curr_node][i]
            if i_min < min:
                min = i_min
                min_node = i
        cost += min
        visited.append(min_node)
        unvisited.remove(min_node)
        curr_node = min_node
    
    cost += matrix[curr_node][start]
    visited.append(start)
    return visited, cost

def two_opt(tour, matrix):
    def calculate_cost(tour, matrix):
        cost = 0
        for i in range(len(tour) - 1):
            cost += matrix[tour[i]][tour[i + 1]]
        return cost
        
    def swap_two_opt(tour, i, k):
        new_tour = tour[:i+1] + tour[i+1:k+1][::-1] + tour[k+1:]
        return new_tour
        
    best_tour = tour
    best_cost = calculate_cost(best_tour, matrix)
        
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for k in range(i + 1, len(tour) - 1):
                new_tour = swap_two_opt(best_tour, i, k)
                new_cost = calculate_cost(new_tour, matrix)
                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    improved = True
        
    return best_tour, best_cost

def NN2O(matrix):
    initial_tour, initial_cost = NN(matrix)
    
    optimized_tour, optimized_cost = two_opt(initial_tour, matrix)
    
    return optimized_tour, optimized_cost

def NN_random(matrix, n_neighbors=3):
    start = random.randint(0, len(matrix) - 1)
    visited = [start]
    unvisited = list(range(len(matrix)))
    unvisited.remove(start)
    curr_node = start
    cost = 0
    
    while unvisited:
        nearest_nodes = sorted(unvisited, key=lambda x: matrix[curr_node][x])
        nearest_nodes = nearest_nodes[:n_neighbors] 
        next_node = random.choice(nearest_nodes)
        
        cost += matrix[curr_node][next_node]
        visited.append(next_node)
        unvisited.remove(next_node)
        curr_node = next_node
    
    cost += matrix[curr_node][start]
    visited.append(start)
    return visited, cost

def RNN(matrix, num_restarts=5, num_nearest=3):
    best_tour = None
    best_cost = float('inf')

    for _ in range(num_restarts):
        initial_tour, initial_cost = NN_random(matrix, num_nearest)

        optimized_tour, optimized_cost = two_opt(initial_tour, matrix)
        
        if optimized_cost < best_cost:
            best_tour = optimized_tour
            best_cost = optimized_cost
    
    return best_tour, best_cost

def calculate_mst_heuristic(adjacency_matrix, unvisited_nodes):
    # Create a graph from the adjacency matrix
    G = nx.Graph()
    
    # Add edges between all unvisited nodes
    for i in unvisited_nodes:
        for j in unvisited_nodes:
            if i != j and adjacency_matrix[i][j] != 0:
                G.add_edge(i, j, weight=adjacency_matrix[i][j])
    
    # Calculate the Minimum Spanning Tree using networkx
    mst = nx.minimum_spanning_tree(G)
    
    # Return the total weight of the MST
    return int(mst.size(weight='weight'))

def A_MST(adjacency_matrix):
    n = len(adjacency_matrix)
    
    # Priority queue: stores (f(n), g(n), current_node, path_taken)
    priority_queue = []
    
    # Initialize the queue with all possible starting nodes
    for start_node in range(n):
        initial_path = [start_node]
        heapq.heappush(priority_queue, (0, 0, start_node, initial_path))
    
    # Visited set is unique to each start path
    best_solution = None
    best_cost = float('inf')

    while priority_queue:
        f, g, current_node, path = heapq.heappop(priority_queue)

        if g >= best_cost:
            continue
        
        # If all nodes are visited, complete the tour and compare with best solution
        if len(path) == n:
            # Add the return to the start node to complete the tour
            total_cost = g + adjacency_matrix[current_node][path[0]]
            if total_cost < best_cost:
                best_cost = total_cost
                best_solution = path + [path[0]]
            continue
        
        # Get the set of unvisited nodes
        unvisited_nodes = set(range(n)) - set(path)
        
        for next_node in unvisited_nodes:
            # Calculate the cost to reach next_node
            g_new = g + adjacency_matrix[current_node][next_node]
            
            # Heuristic: MST of the remaining unvisited nodes
            h_new = calculate_mst_heuristic(adjacency_matrix, list(unvisited_nodes - {next_node}))
            
            # Calculate f(n) = g(n) + h(n)
            f_new = g_new + h_new
            
            # Add the new path to the priority queue
            heapq.heappush(priority_queue, (f_new, g_new, next_node, path + [next_node]))
    
    return best_solution, best_cost        




def main():
    parser = argparse.ArgumentParser(description="Convert a text file to a 2D adjacency matrix.")
    
    # Accept the filename as an argument
    parser.add_argument('filename', type=str, help='Path to the text file containing the adjacency matrix.')
    
    args = parser.parse_args()
    
    # Get the absolute path
    file_path = os.path.join(os.getcwd(), args.filename)
    
    if os.path.exists(file_path):
        matrix = turn_txt_into_2d_matrix(file_path)
    else:
        print(f"Error: File '{file_path}' not found.")
    
    print(NN(matrix))
    print(NN2O(matrix))
    print(RNN(matrix, 25, 3))
    print(A_MST(matrix))
   

if __name__ == "__main__":
    main()

    