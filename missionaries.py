from collections import deque

# Define initial and goal states
initial_state = (3, 3, 1)
goal_state = (0, 0, 0)

# Define possible actions
actions = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]

# Define BFS function
def bfs(initial_state, goal_state, actions):
    # Initialize queue and visited set
    queue = deque()
    queue.append(initial_state)
    visited = set()
    visited.add(initial_state)

    # Initialize parent dictionary
    parent = {}
    parent[initial_state] = None

    # Initialize node count
    node_count = 1

    # BFS algorithm
    while queue:
        # Dequeue node from queue
        node = queue.popleft()

        # Check if goal state is reached
        if node == goal_state:
            path = []
            while node:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, node_count

        # Generate child nodes
        for action in actions:
            # Check if action is valid
            if (node[2] == 1 and (node[0] - action[0] >= node[1] - action[1]) and (node[0] - action[0] >= 0) and (node[1] - action[1] >= 0)) or (node[2] == 0 and ((3 - node[0]) - action[0] >= (3 - node[1]) - action[1]) and ((3 - node[0]) - action[0] >= 0) and ((3 - node[1]) - action[1] >= 0)):
                child = (node[0] - action[0], node[1] - action[1], 1 - node[2])
                if child not in visited:
                    visited.add(child)
                    parent[child] = node
                    queue.append(child)
                    node_count += 1

    # No solution found
    return None, node_count

def dfs(initial_state, goal_state, actions):
    # Initialize stack and visited set
    stack = []
    stack.append(initial_state)
    visited = set()
    visited.add(initial_state)

    # Initialize parent dictionary
    parent = {}
    parent[initial_state] = None

    # Initialize node count
    node_count = 1

    # DFS algorithm
    while stack:
        # Pop node from stack
        node = stack.pop()

        # Check if goal state is reached
        if node == goal_state:
            path = []
            while node:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, node_count

        # Generate child nodes
        for action in actions:
            # Check if action is valid
            if (node[2] == 1 and (node[0] - action[0] >= node[1] - action[1]) and (node[0] - action[0] >= 0) and (node[1] - action[1] >= 0)) or (node[2] == 0 and ((3 - node[0]) - action[0] >= (3 - node[1]) - action[1]) and ((3 - node[0]) - action[0] >= 0) and ((3 - node[1]) - action[1] >= 0)):
                # Generate child node
                child = (node[0] - action[0], node[1] - action[1], 1 - node[2])
                # Check if child node has not been visited
                if child not in visited:
                    # Add child node to stack and visited set
                    stack.append(child)
                    visited.add(child)
                    # Add child node to parent dictionary
                    parent[child] = node
                    # Increment node count
                    node_count += 1

    # If goal state is not reachable, return None
    return None, node_count

# solve using BFS
print("Solution using BFS:")
_, bfs_nodes = bfs(initial_state,goal_state,actions)
print("Nodes evaluated:", bfs_nodes)

# solve using DFS
print("Solution using DFS:")
_, dfs_nodes = dfs(initial_state,goal_state,actions)
print("Nodes evaluated:", dfs_nodes)
