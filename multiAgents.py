# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
# Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    legalMoves.remove("Stop")

# Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"
    print '###################################################################'
    print "chosen action:  " + str(legalMoves[chosenIndex])
    print '###################################################################'

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    # ['__doc__', '__eq__', '__hash__', '__init__', '__module__', '__str__', 'data', 'deepCopy', 'generatePacmanSuccessor', 'generateSuccessor', 'getCapsules', 'getFood', 
    # 'getGhostPosition', 'getGhostPositions', 'getGhostState', 'getGhostStates', 'getLegalActions', 'getLegalPacmanActions', 'getNumAgents', 'getNumFood', 'getPacmanPosition', 
    # 'getPacmanState', 'getScore', 'getWalls', 'hasFood', 'hasWall', 'initialize', 'isLose', 'isWin']
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # dir(successorGameState.data.food) =>
    # ['CELLS_PER_INT', '__doc__', '__eq__', '__getitem__', '__hash__', '__init__', '__module__', '__setitem__', '__str__', '_cellIndexToPosition', '_unpackBits', '_unpackInt', 
    # 'asList', 'copy', 'count', 'data', 'deepCopy', 'height', 'packBits', 'shallowCopy', 'width']

    print "pacman pos:  " + str(newPos)
    score = 0

# if the next move is a lose, don't go there
    if successorGameState.isLose():
      print "lose"
      return 0
# if the next move is a win, go there
    if successorGameState.isWin():
      print "win"
      return 99999

# calculate how close the ghosts are
# this is not being used currently
    ghost_closeness = []
    for state in newGhostStates:
      manhattan = abs(state.getPosition()[0] - newPos[0]) + abs(state.getPosition()[1] - newPos[1])
      ghost_closeness.append(manhattan)
      # print "     " + str(state)
      # print "manhattan :  " + str(manhattan)
    closest_ghost_distance = min(ghost_closeness)
    if closest_ghost_distance <= 1:
      return 0

# find the manhattan distance to all food
    food_distances = []
    for food in successorGameState.data.food.asList():
      manhattan = abs(food[0] - newPos[0]) + abs(food[1] - newPos[1])
      food_distances.append(manhattan)

    closest_food = min(food_distances)
    game_score = currentGameState.getScore()
    next_score = successorGameState.getScore()
    if next_score > game_score:
      return 50 + (next_score-game_score)

    print str(newPos) + " -- " + str(successorGameState.data.food.asList())

# cap the score at 50, but weight up to it for how close the next food is
    score = 50/closest_food

    print "--------------"
    print "score: " + str(score)
    print "--------------"
    return score
    # return successorGameState.getScore()

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    if gameState.isWin or self.depth <= 0:
      return self.evaluationFunction
    print self.depth

    for agent_index in range(0,gameState.getNumAgents()):
      action_costs = {}
      for action in gameState.getLegalActions(agent_index):
        next_state = gameState.generateSuccessor(agent_index, action)
        print next_state
    # Recursive call
        cost = getAction(next_state)
    # --------------
      print action_costs
    return Directions.STOP


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def maxVal(self, gameState, depth):
    if depth == 0 or gameState.isLose() or gameState.isWin():
      return self.evaluationFunction(gameState)
    val = []
    actions = gameState.getLegalActions()
    if Directions.STOP in actions:
      actions.remove(Directions.STOP)
      #Get value (minVal)

  def minVal(self, gameState, depth):
    if depth == 0 or gameState.isLose() or gameState.isWin():
      return self.evaluationFunction(gameState)
    #Ghost...get possible value (maxVal)
    

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
   
                     
              
    #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

