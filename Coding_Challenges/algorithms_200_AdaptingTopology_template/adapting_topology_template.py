#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #
    ctrl, targ = cnot.wires
    def find_shortest_path(graph, source, destination): # BFS
        visited = []
        queue = [[source]]
        
        if source == destination:
            raise BaseException('CNOT not valild!!')
        while queue: # Loop to traverse the graph with the help of the queue
            path = queue.pop(0)
            node = path[-1]
            
            if node not in visited: # If node is not visited
                neighbours = graph[node]
                
                for neighbour in neighbours: # Iterate over the neighbours of the node
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    
                    if neighbour == destination: # if the neighbour node is the destination
                        return new_path
                visited.append(node)
    
        # If the nodes are not connected
        raise BaseException('Not Connected')

    shortest_path = find_shortest_path(graph=graph, source=ctrl, destination=targ)
    return 2*(len(shortest_path)-2) # take out source and destination
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
