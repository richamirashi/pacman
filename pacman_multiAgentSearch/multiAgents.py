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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentFoodList = currentGameState.getFood().asList()
        score = float('inf')

        # iterate over all the food locations,
        # calculate the manhattan distance between pacman and food location,
        # and select the minimum value as a score
        for food in currentFoodList:
            score = min(score, manhattanDistance(food,newPos))
            if Directions.STOP in action:
                return float('-inf')

        # iterate over the future states of ghost(s),
        # and check if pacman and ghost locations are same
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            if ghostPos == newPos:
                return float('-inf')

        # reciprocal of score value
        return 1.0/(1.0 + score)

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        "*** YOUR CODE HERE ***"
        # self.miniMax(gameState, initial_depth, agentIndex)
        movesList = self.miniMax(gameState, 0, 0)
        return movesList[0]

    def miniMax(self, gameState, depth, agentIndex):
        # increases the height of a tree.
        # sets agentIndex value to 0 to switch from ghost to pacman agent
        if(agentIndex >= gameState.getNumAgents()):
            depth += 1
            agentIndex = 0
        # check if the current state is a win or lose state, or if the tree is completely explored
        if(gameState.isWin() or gameState.isLose() or depth == self.depth):
            return self.evaluationFunction(gameState)
        # check if the agent is ghost
        elif(agentIndex != 0):
            return self.minimumValue(gameState, depth, agentIndex)
        # the agent is pacman
        else:
            return self.maximumValue(gameState, depth, agentIndex)

    def maximumValue(self, gameState, depth, agentIndex):
        movesList = gameState.getLegalActions(agentIndex)
        maxScoredMove = ["", float('-inf')]

        for move in movesList:
            successorState = gameState.generateSuccessor(agentIndex, move)
            # find the max score from the successor tree
            score = self.miniMax(successorState, depth, agentIndex + 1)
            if type(score) is not list:
                bestScore = score
            else:
                bestScore = score[1]
            if bestScore > maxScoredMove[1]:
                maxScoredMove = [move, bestScore]
        return maxScoredMove

    def minimumValue(self, gameState, depth, agentIndex):
        movesList = gameState.getLegalActions(agentIndex)
        minScoredMove = ["", float('inf')]

        for move in movesList:
            successorState = gameState.generateSuccessor(agentIndex, move)
            # find the min score from the successor tree
            score = self.miniMax(successorState, depth, agentIndex + 1)
            if type(score) is not list:
                worstScore = score
            else:
                worstScore = score[1]
            if worstScore < minScoredMove[1]:
                minScoredMove = [move, worstScore]
        return minScoredMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    """
    maxScore = max(scores)
    chosenIndex = 0
    for index in range(len(scores)):
        if scores[index] == maxScore:
            chosenIndex = index
    return movesList[chosenIndex]

    minScore = min(scores)
    chosenIndex = 0
    for index in range(len(scores)):
        if scores[index] == minScore:
            chosenIndex = index
    return movesList[chosenIndex]

    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # self.miniMax(gameState, initial_depth, agentIndex)
        movesList = self.miniMax(gameState, 0, 0, float('-inf'), float('inf'))
        return movesList[0]

    def miniMax(self, gameState, depth, agentIndex, alpha, beta):
        # increases the height of a tree.
        # sets agentIndex value to 0 to switch from ghost to pacman agent
        if(agentIndex >= gameState.getNumAgents()):
            depth += 1
            agentIndex = 0

        # check if the current state is a win or lose state, or if the tree is completely explored
        if(gameState.isWin() or gameState.isLose() or depth == self.depth):
            return self.evaluationFunction(gameState)
        # if the agent is a ghost
        elif(agentIndex != 0):
            return self.minimumValue(gameState, depth, agentIndex, alpha, beta)
        # the agent is Pacman
        else:
            return self.maximumValue(gameState, depth, agentIndex, alpha, beta)

    def maximumValue(self, gameState, depth, agentIndex, alpha, beta):
        movesList = gameState.getLegalActions(agentIndex)
        maxScoredMove = ["", float('-inf')]

        for move in movesList:
            successorState = gameState.generateSuccessor(agentIndex, move)
            # find the max score from the successor tree
            score = self.miniMax(successorState, depth, agentIndex + 1, alpha, beta)
            if type(score) is not list:
                bestScore = score
            else:
                bestScore = score[1]
            if bestScore > maxScoredMove[1]:
                maxScoredMove = [move, bestScore]
            # if there is a value greater than the passed value
            if bestScore > beta:
                return [move, bestScore]
            # else reset the value of alpha
            alpha = max(alpha, bestScore)
        return maxScoredMove

    def minimumValue(self, gameState, depth, agentIndex, alpha, beta):
        movesList = gameState.getLegalActions(agentIndex)
        minScoredMove = ["", float('inf')]

        for move in movesList:
            successorState = gameState.generateSuccessor(agentIndex, move)
            # find the min score from the successor tree
            score = self.miniMax(successorState, depth, agentIndex + 1, alpha, beta)
            if type(score) is not list:
                worstScore = score
            else:
                worstScore = score[1]
            if worstScore < minScoredMove[1]:
                minScoredMove = [move, worstScore]
            # if there is a value smaller than the passed value
            if worstScore < alpha:
                return [move, worstScore]
            # else reset the value of beta
            beta = min(beta, worstScore)
        return minScoredMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # self.expectiMax(gameState, depth, agentIndex)
        return self.expectiMax(gameState, 0, 0)

    def expectiMax(self, gameState, depth, agentIndex):
        # increases the height of a tree.
        # sets agentIndex value to 0 to switch from ghost to pacman agent
        if(agentIndex >= gameState.getNumAgents()):
            depth += 1
            agentIndex = 0
        # check if the current state is a win or lose state, or if the tree is completely explored
        if (gameState.isWin() or gameState.isLose() or depth == self.depth):
            return self.evaluationFunction(gameState)
        # check if the agent is ghost
        elif (agentIndex != 0):
            return self.expectedValue(gameState, depth, agentIndex)
        # the agent is pacman
        else:
            return self.maximumValue(gameState, depth, agentIndex)

    def maximumValue(self, gameState, depth, agentIndex):
        movesList = gameState.getLegalActions(agentIndex)
        maxScore = float('-inf')
        maxScoredMove = ""

        for move in movesList:
            successorState = gameState.generateSuccessor(agentIndex, move)
            # find the max score from the successor tree
            score = self.expectiMax(successorState, depth, agentIndex + 1)
            if score > maxScore:
                maxScore = score
                maxScoredMove = move
        if depth > 0:
            return maxScore
        return maxScoredMove

    def expectedValue(self, gameState, depth, agentIndex):
        movesList = gameState.getLegalActions(agentIndex)
        expectedScore = 0
        probability = 1.0/len(movesList)

        for move in movesList:
            successorState = gameState.generateSuccessor(agentIndex, move)
            # find the expected score from the successor tree
            score = self.expectiMax(successorState, depth, agentIndex + 1)
            expectedScore += (probability * score)
        return expectedScore

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    score = scoreEvaluationFunction(currentGameState)
    ghostStates = currentGameState.getGhostStates()
    ghostToPacmanDistance = float('inf')
    capsulesStates = currentGameState.getCapsules()
    capsuleToPacmanDistance = float('inf')
    foodPos = currentGameState.getFood().asList()
    foodToPacmanDistance = float('inf')

    """
    My main focus here is to eat food pellets, and to eat a capsule and kill the ghost.
    Eating a capsule and killing the ghost increases the score significantly
    than running away from ghost and eating food pellets.
    There are no scoring opportunities after eating all the food pellets.
    Hence, pacman sometimes keeps waiting for a ghost to come near to it,
    so that pacman can eat a capsul and kill the ghost.
    """

    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        if(pacmanPos == ghostPos):
            return float('-inf')
        ghostToPacmanDistance = min(ghostToPacmanDistance, manhattanDistance(pacmanPos, ghostPos))
    score += 1.0/(1.0 + (ghostToPacmanDistance / (len(ghostStates))))

    for capsule in capsulesStates:
        capsuleToPacmanDistance = min(capsuleToPacmanDistance, manhattanDistance(pacmanPos, capsule))
    score += 1.0/(1.0 + capsuleToPacmanDistance)

    for food in foodPos:
        foodToPacmanDistance = min(foodToPacmanDistance, manhattanDistance(pacmanPos, food))
    score += 1.0/(foodToPacmanDistance)

    return score

# Abbreviation
better = betterEvaluationFunction
