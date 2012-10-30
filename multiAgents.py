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

The code below is provided as a guide. You are welcome to change
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
    print "chosen action: " + str(legalMoves[chosenIndex])
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

    print "pacman pos: " + str(newPos)
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
      # print " " + str(state)
      # print "manhattan : " + str(manhattan)
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
multi-agent searchers. Any methods defined here will be available
to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

You *do not* need to make any changes here, but you can if you want to
add functionality to all your adversarial search agents. Please do not
remove anything, however.

Note: this is an abstract class: one that should not be instantiated. It's
only partially specified, and designed to be extended. Agent (game.py)
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
    
    best_cost = float('-inf')
    next_action = Directions.STOP

    for action in gameState.getLegalActions(0):
      min_val = self.min_value(0, 1, gameState.generateSuccessor(0, action))

      if (min_val > best_cost) and (action != Directions.STOP):
        best_cost = min_val
        next_action = action

    return next_action
                
  
  def max_value(self, depth, agent, state):
    if depth == self.depth:
      return self.evaluationFunction(state)

    else:
      actions = state.getLegalActions(agent)

      if len(actions) > 0:
        best_cost = float('-inf')
      else:
        best_cost = self.evaluationFunction(state)

      for action in actions:
        current_cost = self.min_value(depth, agent+1, state.generateSuccessor(agent, action))
        if current_cost > best_cost:
          best_cost = current_cost

      return best_cost

  def min_value(self, depth, agent, state):
    if depth == self.depth:
      return self.evaluationFunction(state)

    else:
      actions = state.getLegalActions(agent)

      if len(actions) > 0:
        best_cost = float('inf')
      else:
        best_cost = self.evaluationFunction(state)

      for action in actions:
        if agent == (state.getNumAgents() - 1):
          current_cost = self.max_value(depth+1, 0, state.generateSuccessor(agent, action))
          if current_cost < best_cost:
            best_cost = current_cost

        else:
          current_cost = self.min_value(depth, agent+1, state.generateSuccessor(agent, action))
          if current_cost < best_cost:
            best_cost = current_cost
      return best_cost

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
Your minimax agent with alpha-beta pruning (question 3)
"""

  def getAction(self, gameState):
    """
Returns the minimax action using self.depth and self.evaluationFunction
"""
    best_cost = float('-inf')
    next_action = Directions.STOP
    for action in gameState.getLegalActions(0):
      min_val = self.min_value(0, 1, gameState.generateSuccessor(0, action), float('-inf'), float('inf'))
      if (min_val > best_cost) and (action != Directions.STOP):
        best_cost = min_val
        next_action = action
    return next_action
              

  def max_value(self, depth, agent, state, alpha, beta):
    if depth == self.depth:
      return self.evaluationFunction(state)
    else:
      actions = state.getLegalActions(agent)
      if len(actions) > 0:
        best_cost = float('-inf')
      else:
        best_cost = self.evaluationFunction(state)
      for action in actions:
        best_cost = max(best_cost, self.min_value(depth, agent+1, state.generateSuccessor(agent, action), alpha, beta))
        if best_cost >= beta:
          return best_cost
        alpha = max(best_cost, alpha)
      return best_cost

  def min_value(self, depth, agent, state, alpha, beta):
    if depth == self.depth:
      return self.evaluationFunction(state)
    else:
      actions = state.getLegalActions(agent)
      if len(actions) > 0:
        best_cost = float('inf')
      else:
        best_cost = self.evaluationFunction(state)
      for action in actions:
        if agent == (state.getNumAgents() - 1):
          best_cost = min(best_cost, self.max_value(depth+1, 0, state.generateSuccessor(agent, action), alpha, beta))
          if best_cost <= alpha:
            return best_cost
          beta = min(best_cost, beta)
        else:
          best_cost = min(best_cost, self.min_value(depth, agent+1, state.generateSuccessor(agent, action), alpha, beta))
          if best_cost <= alpha:
            return best_cost
          beta = min(best_cost, beta)
      return best_cost

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    v = float('-inf')
    nextAction = Directions.STOP
    actions = gameState.getLegalPacmanActions()

    for action in actions:
      expVal = self.expVal(0, 1, gameState.generatePacmanSuccessor(action))

      if (expVal > v) and (action != Directions.STOP):
        v = expVal
        nextAction = action

    return nextAction
              

  def maxVal(self, depth, gameState):
    #removed agent from the function head, called PacmanActions
    #removed else statement after this
    if depth == self.depth:
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalPacmanActions()

    if len(actions) > 0:
      v = float('-inf')
    else:
      v = self.evaluationFunction(gameState)

    for action in actions:
      v = max(v, self.expVal(depth, 1, gameState.generatePacmanSuccessor(action)))
                          
    return v

  def expVal(self, depth, agent, gameState):
    #added gameState.isLose()...
    if depth == self.depth or gameState.isLose() or gameState.isWin():
      return self.evaluationFunction(gameState)

    v = 0;
    agentNum = gameState.getNumAgents() - 1
    actions = gameState.getLegalActions(agent)

    for action in actions:
      if agent == (agentNum):
        v += self.maxVal(depth + 1, gameState.generateSuccessor(agent, action))
      else:
        v += self.expVal(depth, agent + 1, gameState.generateSuccessor(agent, action))
                   
    #if len(actions) != 0:
    return v / len(actions)
    #else:
      #return self.evaluationFunction(gameState)



def betterEvaluationFunction(currentGameState):
  """
Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
evaluation function (question 5).

DESCRIPTION: <write something here so we know what you did>
"""
  "*** YOUR CODE HERE ***"
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  new_food_list = newFood.asList()
 
  food_distance = 0
  if len(new_food_list) > 0:
    food_distance = reduce(lambda x, y: x + y, [manhattanDistance(f,newPos) for f in new_food_list])
  
  score = 0
  if len(new_food_list) == 0:
    score = 1000000000
  
  ghost_score = 0
  if (len(newScaredTimes) > 1) and (len(newScaredTimes) < currentGameState.getNumAgents() - 2):
    ghost_score += 100.0
  for ghost_state in newGhostStates:
    distance = manhattanDistance(newPos, ghost_state.getPosition())
    if (ghost_state.scaredTimer == 0) and (distance < 3):
      ghost_score -= 1.0 / (10.0 - distance);
  
    score += 1.0 / (1 + len(new_food_list)) + 1.0 / (1 + food_distance) + ghost_score + currentGameState.getScore()
    
  return score;

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
Your agent for the mini-contest
"""

  def getAction(self, gameState):
    """
Returns an action. You can use any method you want and search to any depth you want.
Just remember that the mini-contest is timed, so you have to trade off speed and computation.

Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
just make a beeline straight towards Pacman (or away from him if they're scared!)
"""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()