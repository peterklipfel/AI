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

  def in_visited(visited, child):
    if not visited:
      return False
    if child[3] == []:
      return False
    childs_parent = child[3].get()
    for node in visited:
      if node[3]:
        nodes_parent = node[3].get()
        if child[0] == node[0] and child[1] == node[1] and child[2] == node[2] and childs_parent[0] == nodes_parent[0] and childs_parent[1] == nodes_parent[1] and childs_parent[2] == nodes_parent[2]:
          return True

    return False

  frontier = []
  visited = []
  start = problem.getStartState()
  start = [start, "none", 0, []]

  frontier.insert(0, start)

  while frontier:
    child_to_parent = {}
    current_node = frontier.pop()
    visited.append(current_node)
    
    if problem.isGoalState(current_node[0]):
      directions = []
      path_node = current_node
      directions.insert(0, path_node[1])
      while path_node != start:
        path_node = path_node[3].get()
        directions.insert(0,path_node[1])
      directions.pop(0) # this is to get rid of the start "none" direction
      print directions
      return directions

    children = problem.getSuccessors(current_node[0])
    for child in children:
      child = [child[0], child[1], child[2], ref(current_node)]
      if not in_visited(visited, child):
        frontier.insert(0, child)


  # def at_goal(current_depth_nodes):
  #   if not current_depth_nodes: 
  #     print "nothing in current depth"
  #     return True
  #   for position in current_depth_nodes:
  #     if problem.isGoalState(position[0]):
  #       return True
  #   return False


  # start = problem.getStartState()
  # iterative_depth_array = []
  # current_depth_nodes = []
  # children = problem.getSuccessors(start)
  # for child in children:
  #   current_depth_nodes.append(child)

# iterative_depth_array is a collection of the frontiers.  It
# can be imagined as each depth of the search tree
  iterative_depth_array.append(current_depth_nodes)

  # while not at_goal(current_depth_nodes):
  #   next_depth_nodes = []
  #   for node in current_depth_nodes:
  #     children_will_be_added = True
  #     children = problem.getSuccessors(node[0])
  #     for a_depth in iterative_depth_array:
  #       for child in children:
  #         if child in a_depth:
  #           children_will_be_added = False
  #     if children_will_be_added:
  #       for child in children:
  #         next_depth_nodes.append(child)
  #   current_depth_nodes = next_depth_nodes
  #   if next_depth_nodes:
  #     iterative_depth_array.append(current_depth_nodes)

# this section of the code is walking back up the stored frontiers,
# and putting the path that it finds into the 'directions' queue.
# This works because before putting a node onto the next frontier
# the algorithm ensures that the node doesn't exist elsewhere in the
# gathered frontiers.  This means there will be no conflicts

  # goal_position = []
  # for node in iterative_depth_array[-1]:
  #   if problem.isGoalState(node[0]):
  #     goal_position = node

  # current_position = goal_position
  # directions = []
  # iterative_depth_array.pop()
  # if goal_position:
  #   directions.insert(0,goal_position[1])

  # while iterative_depth_array:
  #   next_depth_nodes = iterative_depth_array.pop()
  #   for node in next_depth_nodes:
  #     if current_position:
  #       children = problem.getSuccessors(current_position[0])
  #       for child in children: 
  #         if node[0] == child[0] :
  #           directions.insert(0,node[1])
  #           current_position = node
  # print directions
  # return directions
      
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
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  e = Directions.EAST
  n = Directions.NORTH
  distance = 0
  ##Direction, Parent, Calculated heuristic, children, distance
  enodes = {}
  ngs = []
  start = problem.getStartState()
  current = start
  enodes[current] = ['Startoo','Startoo', -1, [], distance]
  ngs += ['Startoo']

  while True:
    expanded = problem.getSuccessors(current)
    ##Increment distance for the algorithm
    distance = distance + 1
    ##Sets it to the program knows this node is expanded, will prevent any double expansions
    enodes[current][2] = -1

    ##Format the enodes dictionary and add the children nodes as well as various other information
    for nodes in expanded:
      if nodes[0] not in enodes:
        enodes[nodes[0]] = [nodes[1], current, distance + heuristic(nodes[0], problem), [], distance]
        enodes[current][3] += [nodes[0]]

    ##Checks the nodes to see if they are the goal state, if not then expand another node
    for nodes in enodes:
      if nodes not in ngs:
        if problem.isGoalState(nodes):
          ##Generate the path from the goal node to the start node
          temp = []
          path = []
          while True:
            temp = enodes[nodes][1]
            if(enodes[nodes][0] != 'Startoo'):
              path.append(enodes[nodes][0])
            else:
              break
            nodes = temp

          path.reverse()
          return path
        else:
          ngs += [nodes]

    ##Get the best node for expansion
    ##Temporary variable to store the heuristic size
    temp = 0
    for nodes in enodes:
      if enodes[nodes][2] != -1:
        if temp == 0:
          temp = enodes[nodes][2]
          current = nodes
          distance = enodes[nodes][4]
        elif enodes[nodes][2] < temp:
          temp = enodes[nodes][2]
          current = nodes
          distance = enodes[nodes][4]

  return [w]

    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch