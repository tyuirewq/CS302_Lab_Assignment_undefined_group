from queue import Queue
from collections import deque
import time

# define initial and goal states
initial_state = ("W", "W", "W", "-", "E", "E", "E")
goal_state = ("E", "E", "E", "-", "W", "W", "W")

# define operator to move forward
def move_forward(state, i, j):
    if state[i] == "W" and state[j] == "-":
        return state[:i] + ("-",) + state[i+1:j] + ("W",) + state[j+1:]
    elif state[i] == "-" and state[j] == "E":
        return state[:i] + ("E",) + state[i+1:j] + ("-",) + state[j+1:]
    return None

# define DFS function
def dfs(initial_state, goal_state):
    stack = [(initial_state, [])]
    visited = set()
    nodes_evaluated = 0
    start_time = time.time()

    while stack:
        state, path = stack.pop()
        nodes_evaluated += 1

        if state == goal_state:
            end_time = time.time()
            print("DFS solution found in {:.4f} seconds with {} nodes evaluated".format(end_time - start_time, nodes_evaluated))
            return path + [state]

        visited.add(state)

        for i in range(7):
            for j in range(i+1, 7):
                child_state = move_forward(state, i, j)
                if child_state is not None and child_state not in visited:
                    stack.append((child_state, path + [state]))

    end_time = time.time()
    print("DFS solution not found in {:.4f} seconds with {} nodes evaluated".format(end_time - start_time, nodes_evaluated))
    return None

# define BFS function
def bfs(initial_state, goal_state):
    queue = deque([(initial_state, [])])
    visited = set()
    nodes_evaluated = 0
    start_time = time.time()

    while queue:
        state, path = queue.popleft()
        nodes_evaluated += 1

        if state == goal_state:
            end_time = time.time()
            print("BFS solution found in {:.4f} seconds with {} nodes evaluated".format(end_time - start_time, nodes_evaluated))
            return path + [state]

        visited.add(state)

        for i in range(7):
            for j in range(i+1, 7):
                child_state = move_forward(state, i, j)
                if child_state is not None and child_state not in visited:
                    queue.append((child_state, path + [state]))

    end_time = time.time()
    print("BFS solution not found in {:.4f} seconds with {} nodes evaluated".format(end_time - start_time, nodes_evaluated))
    return None

# test BFS function
bfs(initial_state, goal_state)

# test DFS function
dfs(initial_state, goal_state)
