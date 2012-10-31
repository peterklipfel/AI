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
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    score = 0

  # if the next move is a lose, don't go there
    if successorGameState.isLose():
      return 0
  # if the next move is a win, go there
    if successorGameState.isWin():
      return 99999

  # find the manhattan distance to all food
    food_distances = successorGameState.data.food.asList()
    def find_manhattan(food):
      return abs(food[0] - newPos[0]) + abs(food[1] - newPos[1])
    closest_food = min(map(find_manhattan, food_distances))

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

  # cap the score at 50, but weight up to it for how close the next food is
    score = 50/closest_food

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
    
    bestCost = float('-inf')
    nextAction = Directions.STOP

    for action in gameState.getLegalActions(0):
      gameStates = gameState.generatePacmanSuccessor(action)
      minVal = self.minVal(0, 1, gameStates)
      
      if (minVal > bestCost) and (action != Directions.STOP):
        bestCost = minVal
        nextAction = action

    return nextAction
                
  
  def maxVal(self, depth, gameState):
    if depth == self.depth:
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalPacmanActions()
    
    if len(actions) > 0:
      bestCost = float('-inf')
    else:
      bestCost = self.evaluationFunction(gameState)

    for action in actions:
      gameStates = gameState.generatePacmanSuccessor(action)
      currentCost = self.minVal(depth, 1, gameStates)
      if currentCost > bestCost:
        bestCost = currentCost

    return bestCost

  def minVal(self, depth, agent, gameState):
    if depth == self.depth:
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agent)
    numAgents = gameState.getNumAgents() - 1

    if len(actions) > 0:
      bestCost = float('inf')
    else:
      bestCost = self.evaluationFunction(gameState)

    for action in actions:
      gameStates = gameState.generateSuccessor(agent, action)
      if agent == (numAgents):
        currentCost = self.maxVal(depth+1, gameStates)
        if currentCost < bestCost:
          bestCost = currentCost

      else:
        currentCost = self.minVal(depth, agent+1, gameStates)
        if currentCost < bestCost:
          bestCost = currentCost
    return bestCost

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
Returns the minimax action using self.depth and self.evaluationFunction
"""
    bestCost = float('-inf')
    nextAction = Directions.STOP
    actions = gameState.getLegalPacmanActions()
    if Directions.STOP in actions:
      actions.remove(Directions.STOP)
    for action in actions:
      minVal = self.minVal(0, 1, gameState.generatePacmanSuccessor(action), float('-inf'), float('inf'))
      if (minVal > bestCost):
        bestCost = minVal
        nextAction = action
    return nextAction
              

  def maxVal(self, depth, gameState, alpha, beta):
    if depth == self.depth:
      return self.evaluationFunction(gameState)
    actions = gameState.getLegalPacmanActions()
    if len(actions) > 0:
      bestCost = float('-inf')
    else:
      bestCost = self.evaluationFunction(gameState)
    for action in actions:
      bestCost = max(bestCost, self.minVal(depth, 1, gameState.generatePacmanSuccessor(action), alpha, beta))
      if bestCost >= beta:
        return bestCost
      alpha = max(bestCost, alpha)
    return bestCost

  def minVal(self, depth, agent, gameState, alpha, beta):
    if depth == self.depth:
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agent)
    numAgents = gameState.getNumAgents() - 1

    if len(actions) > 0:
      bestCost = float('inf')
    else:
      bestCost = self.evaluationFunction(gameState)
    for action in actions:
      gameStates = gameState.generateSuccessor(agent,action)
      if agent == (numAgents):
        bestCost = min(bestCost, self.maxVal(depth + 1, gameStates, alpha, beta))
        if bestCost <= alpha:
          return bestCost
        beta = min(bestCost, beta)
      else:
        bestCost = min(bestCost, self.minVal(depth, agent + 1, gameStates, alpha, beta))
        if bestCost <= alpha:
          return bestCost
        beta = min(bestCost, beta)
    return bestCost

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

    DESCRIPTION:
    variables
    -----------------------------------------------------------------------------------------------
    new_pos          -> Pac man's new position
    new_food         -> the new state of the food
    new_ghost_states -> the new state of the ghosts
    new_scared_times -> a check of whether ghosts are scared
    new_food_list    -> the new state of the food in list form
    food_distance    -> the sum of all the manhattan distances
    score            -> utility of the state we are checking
    ghost_score      -> how far away from ghost we want to be
    -----------------------------------------------------------------------------------------------
    explanation
    -----------------------------------------------------------------------------------------------
    We first calculate food_distance to be the sum of the manhattan distances to every piece of 
    food.  The intention of this is to move pacman in the general direction of food and try to 
    minimize the seperation of food pieces.  
    
    Next, we check to see if there is no food left.  If ther is none, then we return an arbitrarily
    large number (integer to retain performance) to indicate a win state.  
    
    The next calculation is to account for the opponent ghosts.  The first check is to see whether
    the scared times will last to the next move.  If they do, then we start the score with a bonus.
    Then we iterate over all the states and subtract out the score from the ghosts based on how 
    close they are.  I chose `3` because it seemed to make a more daring pacman and living on the
    edge is better.  Originally, I was considering `2`, but decided against it because pacman has a
    lower likelyhood of being eaten if he looks ahead 2 squares instead of just 1.  Though 5 felt
    a bit too conservative, and 4 didn't seem to do quite as well when I ran the suites many times. 

    Finally, we add a weight to the score we return based on the state of each ghost.  In the 
    numerator we have 1/(number of food pieces left), so this weights the score up as the number of
    food pieces decreases.  In the denominator, we have food_distance.  Again, this weights the
    score up when the general distance to food decreases.  Finally, we add the score of the current
    ghost (which will be negative unless the ghost is scared) and then add the score that the API
    provides.  This is to help push Pacman toward food, and toward scared ghosts. 
    ----------------------------------------------------------------------------------------------- 
  """
  "*** YOUR CODE HERE ***"
  new_pos = currentGameState.getPacmanPosition()
  new_food = currentGameState.getFood()
  new_ghost_states = currentGameState.getGhostStates()
  new_scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]

  new_food_list = new_food.asList()
 
  food_distance = 0
  if len(new_food_list) > 0:
    # http://docs.python.org/2/library/functions.html#reduce ... Thanks Professor Chang
    food_distance = reduce(lambda x, y: x + y, [manhattanDistance(f,new_pos) for f in new_food_list])
  
  score = 0
  if len(new_food_list) == 0:
    score = 99999999
  
  ghost_score = 0
  if (len(new_scared_times) > 1):
    ghost_score += 100.0
  for ghost_state in new_ghost_states:
    distance = manhattanDistance(new_pos, ghost_state.getPosition())
    if (ghost_state.scaredTimer == 0) and (distance < 3):
      ghost_score -= 1.0 / (10.0 - distance);
  
    score += ((1.0 / (1 + len(new_food_list)) + 1.0) / (1 + food_distance)) + ghost_score + currentGameState.getScore()
    
  return score;

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
    # "*** YOUR CODE HERE ***"
    score = -999999
    best_action = None
    actions = gameState.getLegalActions()
    actions.remove("Stop")
    for action in actions:
      next_state = gameState.generateSuccessor(0, action)
      current_score = betterEvaluationFunction(next_state)
      if current_score > score:
        score = current_score
        best_action = action
    if best_action == None:
      best_action = Directions.STOP
    return best_action


