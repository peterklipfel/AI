# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class ref:
  def __init__(self, obj): self.obj = obj
  def get(self):    return self.obj
  def set(self, obj):      self.obj = obj


class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"

  # print "Start:", problem.getStartState()
  # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  # print "Start's successors", problem.getSuccessors(problem.getStartState())
  # print "Second successors:", problem.getSuccessors(problem.getSuccessors(problem.getStartState())[0][0])
  # print dir(problem)

  # Every list item is of the form
  # {(x, y), Direction from parent, cost?}


  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  e = Directions.EAST
  n = Directions.NORTH
  start = problem.getStartState()
  children = problem.getSuccessors(start)
  will_search = []
  visited_nodes = []
  parent_nodes = {}
 
  children = [problem.getSuccessors(start)]
 
  ##Sets it so the current children will be searched and sets their parent in a dictionary
  for child in children:
    for childs in child:
      will_search.append(childs)
      parent_nodes[childs[0]] = start;
 
     
  while will_search:
    ##Assign a new current node based on what needs to be searched
    current_node = will_search.pop()
       
    ##If the current node is the goal, create a path and return it
    if problem.isGoalState(current_node[0]):
      ##Directions is the list to be returned and temp is a temporary storage for a node
      directions = []
      temp = []
 
      temp = current_node[0]
      current_node = current_node[0]
      ##Calculates a path based on the current node and it's parent
      while (current_node != start):
        temp = parent_nodes[temp]
        if(temp[0] - current_node[0] > 0):
          directions.append(w)
        elif(temp[0] - current_node[0] < 0):
          directions.append(e)
        elif(temp[1] - current_node[1] > 0):
          directions.append(s)
        elif(temp[1] - current_node[1] < 0):
          directions.append(n)
 
        current_node = temp;
 
      directions.reverse()
      return directionst
       
    ##Add the current node to the list of nodes visited
    visited_nodes.append(current_node)
    ##Expand the current node and store it's children
    children = [problem.getSuccessors(current_node[0])]
    ##Check nodes in the lists and add them to the appropriate places
    for child in children:
      for childs in child:
        if childs not in will_search and childs not in visited_nodes:
          will_search.append(childs)
        if childs[0] not in parent_nodes and childs[0] not in visited_nodes:
          parent_nodes[childs[0]] = current_node[0]

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  "*** YOUR CODE HERE ***"

  frontier = []
  visited = []
  start = problem.getStartState()
  # [(int, int), "string", int]
  start = (start, "none", 0)

  frontier.insert(0, start)

  child_to_parent = {}
  while frontier:
    current_node = frontier.pop()
    visited.append(current_node)
    
    children = problem.getSuccessors(current_node[0])
    for child in children:
      if child not in visited:
        # print "children"
        # print children
        # print "child"
        # print child
        child_to_parent[child] = current_node
        frontier.insert(0, child)
      
    if problem.isGoalState(current_node[0]):
      print "Found it"
      directions = []
      path_node = current_node
      directions.insert(0, path_node[1])
      print current_node
      while path_node != start:
        new_node = child_to_parent[path_node]
        path_node = new_node
        directions.insert(0,path_node[1])
      directions.pop(0) # this is to get rid of the start "none" direction
      print directions
      return directions

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  # def get_min_node(frontier):
  #   min_node = frontier[0]
  #   for node in frontier:
  #     if node[2] < min_node:
  #       min_node = node
  #   return min_node

  # print problem.getSuccessors(problem.getStartState())
  start = problem.getStartState()
  children = problem.getSuccessors(start)
  frontier = util.PriorityQueue()
  visited = []
  for child in children:
    frontier.push(child, child[2])
  current_path = []
  while True:
    if not frontier:
      print "No Solution"
      return []
    current_node = frontier.pop()
    current_path.append(current_node)
    visited.append(current_node)

    if problem.isGoalState(current_node[0]):
      # print "found solution"
      # print current_path
      directions = []
      for node in current_path:
        directions.append(node[1])
      return directions
    visited.append(current_node)
    children = problem.getSuccessors(current_node[0])

    for child in children:
      if child not in visited:
        frontier.push(child, child[2])




def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  distance = 0
  ##enodes keeps track of all the information of a given node
  ##Direction, Parent, Calculated heuristic, children, distance
  enodes = {}
  ##Checks to see what is and isn't a goalstate
  start = problem.getStartState()
  current = start
  enodes[current] = ['Startoo','Startoo', -1, [], distance]
  toCheck = []
  nlist = []
  nlist.append(current)

  while True:
    expanded = problem.getSuccessors(current)
    ##Increment distance for the algorithm
    distance = distance + 1
    ##Sets it to the program knows this node is expanded, will prevent any double expansions
    enodes[current][2] = -1

    ##Format the enodes dictionary and add the children nodes as well as various other information
    for nodes in expanded:
      if nodes[0] not in nlist:
        nlist.append(current)
        enodes[nodes[0]] = [nodes[1], current, distance + heuristic(nodes[0], problem), [], distance]
        enodes[current][3] += [nodes[0]]
        toCheck += [nodes[0]]

    ##Checks the nodes to see if they are the goal state, if not then expand another node
    for nodes in toCheck:
      nodes = toCheck.pop()
      #if nodes not in ngs:
      if problem.isGoalState(nodes):
         ##Generate the path from the goal node to the start node
        temp = []
        path = []
        while True:
          temp = enodes[nodes]
          if(temp[0] != 'Startoo'):
            path.append(temp[0])
          else:
            break
          nodes = temp[1]

        path.reverse()
        return path
        #else:
        #  ngs += [nodes]

    ##Get the best node for expansion
    ##Temporary variable to store the heuristic size
    temp = 0
    for nodes, stuff in enodes.iteritems():
      if(stuff[2] != -1):
        if temp == 0:
          temp = stuff[2]
          current = nodes
          distance = stuff[4]
        elif stuff[2] < temp:
          temp = stuff[2]
          current = nodes
          distance = stuff[4]

  return [w]
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch