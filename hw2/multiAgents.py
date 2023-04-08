# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def chebyshev_distance(self, p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        return max(abs(x2 - x1), abs(y2 - y1))

    def has_nonzero(self, lst):
        """
        Check if a list contains any non-zero elements.
        """
        for num in lst:
            if num != 0:
                return True
        return False

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # print(chosenIndex, legalMoves[chosenIndex])
        # Maybe not pick random, pick "stop"?
        # print("="*10)
        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def get_food(self, curGameState: GameState, nextGameState: GameState):
        """
        Check if the agent is eating food.
        """
        curFood = curGameState.getFood()
        nextFood = nextGameState.getFood()
        # print(curFood.asList(), nextFood.asList())
        if curFood.asList() != nextFood.asList():
            return True
        return False

    def get_capsule(self, curGameState: GameState, nextGameState: GameState):
        """
        Check if the agent is eating capsule.
        """
        curCapsule = curGameState.getCapsules()
        nextCapsule = nextGameState.getCapsules()
        # print(curCapsule, nextCapsule)
        if curCapsule != nextCapsule:
            return True
        return False

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition() #(x, y) in tuple
        newFood = successorGameState.getFood() #current food position, Grid of boolean food indicator variables
        # 10 points for every food you eat
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.getFood()
        if currentFood[x][y] == True: ...
        """

        newCapsule = successorGameState.getCapsules()
        # 200 points for every ghost you eat
        # but no point for capsule

        # For Ghost
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # scare time for each ghost
        # Position of ghost do not change regardless of your state
        # because you can't predict the future
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates] # list of tuples
        # Count down from 40 moves
        ghostStartPos = [ghostState.start.getPosition() for ghostState in newGhostStates] # always the same for a ghost, list of tuples

        dist_pac_ghost = [manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]

        # give higher score if this action makes pacman further away from ghost
        dist_ghost = sum(dist_pac_ghost)

        dist_pac_capsule = [manhattanDistance(newPos, capsulePos) for capsulePos in newCapsule]
        
        dist_closest_capsule = min(dist_pac_capsule) if dist_pac_capsule else 0
        
        dist_pac_food = [
                manhattanDistance(newPos, foodPos) for foodPos 
                in newFood.asList() # list of tuples of food position
            ]
        
        dist_closest_food = min(dist_pac_food) if dist_pac_food else 0

        # Normal mode
        if self.has_nonzero(newScaredTimes) == False:
            # if the ghost is far away, eat as much capsule and food as possible
            if dist_ghost > 5:
                # get capsule first
                if self.get_capsule(currentGameState, successorGameState): # agent eats capsule
                    ttl_score = 1000000
                # get food second
                if self.get_food(currentGameState, successorGameState): # agent eats food
                    ttl_score = 1000000

                # couldn't get any food, go to the closest capsule and food
                else:
                    ttl_score = -50*dist_closest_capsule - dist_closest_food
            # if the ghost is close, eat as much capsule as possible
            else:
                ttl_score = 5*dist_ghost - 12*dist_closest_capsule - 4*dist_closest_food
        
        # Scary mode
        else:
            ttl_score =- dist_ghost
        # print(action, -5*dist_closest_capsule, - dist_closest_food)
        return ttl_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
           
        best_v, best_action = self.minimax(gameState, depth=0, agentIndex=0)
        return best_action
    
    def minimax(self, gameState: GameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0:
            # If the current agent is Pacman, find the maximum score and action
            # among Pacman's possible actions.
            best_score = float('-inf')
            best_action = None
            for action in gameState.getLegalActions(agentIndex):
                successor_state = gameState.generateSuccessor(agentIndex, action)
                score, _ = self.minimax(
                    successor_state, 
                    depth, 
                    (agentIndex + 1) % gameState.getNumAgents()
                    )
                if score > best_score:
                    best_score = score
                    best_action = action
        else:
            # If the current agent is a ghost, find the minimum score and action
            # among the ghost's possible actions.
            best_score = float('inf')
            best_action = None
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    # If this is the last ghost, decrement the depth and move on to Pacman.
                    score, _ = self.minimax(successorState, depth + 1, 0)
                else:
                    # Otherwise, move on to the next ghost.
                   score, _ = self.minimax(successorState, depth, agentIndex+1)
                if score < best_score:
                    best_score = score
                    best_action = action
        return best_score, best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        best_v, best_action = self.minimaxWithAlphaBetaPruning(
            gameState, depth=0, agentIndex=0, 
            alpha=float('-inf'), beta=float('inf')
            )
        return best_action

    def minimaxWithAlphaBetaPruning(self, gameState: GameState, depth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0:
            # If the current agent is Pacman, find the maximum score and action
            # among Pacman's possible actions.
            best_score = float('-inf')
            best_action = None
            for action in gameState.getLegalActions(agentIndex):
                successor_state = gameState.generateSuccessor(agentIndex, action)
                score, _ = self.minimaxWithAlphaBetaPruning(
                    successor_state, depth, 
                    (agentIndex + 1) % gameState.getNumAgents(),
                    alpha, beta
                    )
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if beta < alpha:
                    break
        else:
            # If the current agent is a ghost, find the minimum score and action
            # among the ghost's possible actions.
            best_score = float('inf')
            best_action = None
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    # If this is the last ghost, increment the depth and move on to Pacman.
                    score, _ = self.minimaxWithAlphaBetaPruning(successorState, depth + 1, 0, alpha, beta)
                else:
                    # Otherwise, move on to the next ghost.
                   score, _ = self.minimaxWithAlphaBetaPruning(successorState, depth, agentIndex+1, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, best_score)
                if beta < alpha:
                    break
        return best_score, best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        best_v, best_action = self.expectimax(gameState, depth=0, agentIndex=0)
        return best_action

    def expectimax(self, gameState: GameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0:
            # If the current agent is Pacman, find the maximum score and action
            # among Pacman's possible actions.
            best_score = float('-inf')
            best_action = None
            for action in gameState.getLegalActions(agentIndex):
                successor_state = gameState.generateSuccessor(agentIndex, action)
                score, _ = self.expectimax(
                    successor_state, 
                    depth, 
                    (agentIndex + 1) % gameState.getNumAgents()
                    )
                if score > best_score:
                    best_score = score
                    best_action = action
        else:
            # If the current agent is a ghost, find the average score and action
            # among the ghost's possible actions.
            best_score = 0
            best_action = None
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    # If this is the last ghost, increment the depth and move on to Pacman.
                    score, _ = self.expectimax(successorState, depth + 1, 0)
                else:
                    # Otherwise, move on to the next ghost.
                   score, _ = self.expectimax(successorState, depth, agentIndex+1)
                best_score += score
            best_score /= len(gameState.getLegalActions(agentIndex))
        return best_score, best_action