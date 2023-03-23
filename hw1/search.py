# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    stack = util.Stack()
    visited = set()
    # initialize a dictionary to keep track of the parent vertex of each vertex 
    start = problem.getStartState()

    parents = dict()
    parents = {
        start: [None, None] # current node :[parent node position, direction (parent->current)]
        }
    stack.push(start)

    if problem.isGoalState(start):
        return []
    
    else:
        while not stack.isEmpty():
            cur_node_pos = stack.pop() # the first node only stores position info
            # print("current node:", cur_node_pos)
            if cur_node_pos not in visited:
                visited.add(cur_node_pos)

            # check if it's the goal
            if problem.isGoalState(cur_node_pos):
                path = []
                while cur_node_pos is not None:
                    # track back to the first node (the parent node of the first node is none)
                    prev_node_dir = parents[cur_node_pos][1]
                    path.append(prev_node_dir)
                    cur_node_pos = parents[cur_node_pos][0]
                    
                path = path[::-1][1:] # reverse directions and remove None
                return path
            
            # else search
            neighbors = problem.getSuccessors(cur_node_pos)
            for neighbor in neighbors:
                neighbor_pos = neighbor[0]
                par_neigh_dir = neighbor[1]
                neighbor_cost = neighbor[2]

                if neighbor_pos not in visited:
                    stack.push(neighbor_pos)
                    parents[neighbor_pos] = [cur_node_pos, par_neigh_dir]
                           
    return []
    

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = set()
    # initialize a dictionary to keep track of the parent vertex of each vertex 
    start = problem.getStartState()

    parents = dict()
    parents = {
        start: [None, None] # current node :[parent node position, direction (parent->current)]
        }

    if problem.isGoalState(start):
        return []
    
    queue.push(start)
    visited.add(start)

    while not queue.isEmpty():
        cur_node_pos = queue.pop()

        if problem.isGoalState(cur_node_pos):
            end_pos = cur_node_pos
            break
        neighbors = problem.getSuccessors(cur_node_pos)
        for neighbor in neighbors:
            neighbor_pos = neighbor[0]
            par_neigh_dir = neighbor[1]

            if neighbor_pos not in visited:
                visited.add(neighbor_pos)
                parents[neighbor_pos] = [cur_node_pos, par_neigh_dir]
                queue.push(neighbor_pos)

    path = []
    if cur_node_pos in visited:
        while end_pos != start:
            prev_node_dir = parents[end_pos][1]
            path.append(prev_node_dir)
            end_pos = parents[end_pos][0]
            
        path = path[::-1] # reverse directions
        return path  
    
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()

    start = problem.getStartState() # the first node only stores position info

    parents = {
        start: [None, None] # current node :[parent node position, direction (parent->current)]
        }

    explored = {
        start: 0 #explored node's position, path cost (from starting node to the node)
    }

    pq.push(start, priority=0)

    if problem.isGoalState(start):
        return []   
    
    # pos, neighbor direction, neighbor cost
    while not pq.isEmpty():
        cur_node_pos = pq.pop()
        cur_node_cost = explored[cur_node_pos]

        if problem.isGoalState(cur_node_pos):
            path = []
            while cur_node_pos is not None:
                # track back to the first node (the parent node of the first node is none)

                prev_node_dir = parents[cur_node_pos][1]
                path.append(prev_node_dir)
                cur_node_pos = parents[cur_node_pos][0]
            path = path[::-1][1:] # reverse directions and remove None
            return path
            
        neighbors = problem.getSuccessors(cur_node_pos)

        for neighbor in neighbors:
            neighbor_pos = neighbor[0]
            par_neigh_dir = neighbor[1]
            neighbor_cost = neighbor[2]

            neigh_path_cost = cur_node_cost + neighbor_cost

            if neighbor_pos not in explored:
                explored[neighbor_pos] = neigh_path_cost
                parents[neighbor_pos] = [cur_node_pos, par_neigh_dir]
                pq.push(neighbor_pos, neigh_path_cost)

            elif neigh_path_cost < explored[neighbor_pos]:
                explored[neighbor_pos] = neigh_path_cost
                parents[neighbor_pos] = [cur_node_pos, par_neigh_dir]
                pq.update(neighbor_pos, neigh_path_cost)

    return None
        

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    start = problem.getStartState() # the first node only stores position info    

    parents = {
        start: [None, None] # current node :[parent node position, direction (parent->current)]
        }

    explored = {
        start: 0 #explored node's position, path cost (from starting node to the node)
    }

    pq.push(start, priority=0)

    if problem.isGoalState(start):
        return []   

    # pos, neighbor direction, neighbor cost
    while not pq.isEmpty():
        cur_node_pos = pq.pop()
        cur_node_cost = explored[cur_node_pos]

        if problem.isGoalState(cur_node_pos):
            path = []

            while cur_node_pos is not None:
                # track back to the first node (the parent node of the first node is none)
                prev_node_dir = parents[cur_node_pos][1]
                path.append(prev_node_dir)
                cur_node_pos = parents[cur_node_pos][0]
            path = path[::-1][1:] # reverse directions and remove None
            return path
            
        neighbors = problem.getSuccessors(cur_node_pos)

        for neighbor in neighbors:
            neighbor_pos = neighbor[0]
            par_neigh_dir = neighbor[1]
            neighbor_cost = neighbor[2]
            estimated_cost = heuristic(neighbor_pos, problem)

            neigh_path_cost = cur_node_cost + neighbor_cost

            if neighbor_pos not in explored:
                explored[neighbor_pos] = neigh_path_cost
                parents[neighbor_pos] = [cur_node_pos, par_neigh_dir]
                neigh_path_cost = cur_node_cost + neighbor_cost + estimated_cost
                pq.push(neighbor_pos, neigh_path_cost)

            elif neigh_path_cost < explored[neighbor_pos]:
                explored[neighbor_pos] = neigh_path_cost
                parents[neighbor_pos] = [cur_node_pos, par_neigh_dir]
                neigh_path_cost = cur_node_cost + neighbor_cost + estimated_cost
                pq.update(neighbor_pos, neigh_path_cost)

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
