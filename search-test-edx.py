"""
*p1 search testing code from Alyxi at edx
2012-10-08T22:06:47-0600  - 3 days ago

The auto grader output isn't amazing, and I was having trouble tracking down what was causing the errors. Most of the supplied problems are complicated, and hard to check partway with asserts or print statements.
Since it's a graph search problem, the absolute simplest way to test and debug your function is on a graph. To make your own search problem, all you need to do is define an instance of the SearchProblem class (plus a heuristic for A*).
I've included my own code to do this below, to specify your own graph add a new one similar to agraph - read the string at the start of TestVerySimpleSearch for the format. To specify your own heuristic, add one similar to letterHeuristic. LetterHeuristic assumes that graphs with character names are at least their distance in the alphabet apart (A>C would be at least 2 apart.) Check the forum hasn't messed up the formatting if you use it, too, since python is whitespace dependent.
To test and debug, run the file (python filename.py) and compare the output to what you'd expect. If there's a mismatch, you can use print statements (or asserts) within your search.py code to debug the search functions and see which step your function is going astray on. If you're wondering why all the comments are as """ """, the forum formatting doesn't seem to like hashes.
"""

import search
class TestVerySimpleGraph(search.SearchProblem):
    """Gamestate is a list of tuples [('X', 'Y', z), ('A', 'B', c),...]
A tuple ('X', 'Y', z) represents an edge in a graph between X and Y, costing z amount to traverse between them. Edges are assumed to be non-directed.
Goal is the node label for the goal, and start is the node label for the start.
Thus, to initialize a search problem, use TestVerySimpleSearch(Gamestate,goal,start) with appropriate values for all three."""
    def __init__(self, gameState, goal, start):
        self.startState = start
        self.goal = goal
        self.gameState = gameState

    def getStartState(self):
        return self.startState

    def isGoalState(self,state):
        return state == self.goal

    def getSuccessors(self, state):
        successorList = []
        for edge in self.gameState:
            if edge[0]==state:
                successorList.append((edge[1],edge[0]+edge[1],edge[2]))
            if edge[1]==state:
                successorList.append((edge[0],edge[1]+edge[0],edge[2]))
        return successorList

    def getCostOfActions(self,actions):
        cost=0
        for action in actions:
            for edge in self.gameState:
                if edge[0]==action[0] and edge[1]==action[1]:
                    cost+=edge[2]
                if edge[1]==action[0] and edge[0]==action[1]:
                    cost+=edge[2]
        return cost

    def runTest(self):
        print "Path result for DFS:",search.depthFirstSearch(self)
        print "Path result for BFS:",search.breadthFirstSearch(self)
        print "Path result for UCS:",search.uniformCostSearch(self)
        print "Path result for A*:",search.aStarSearch(self,search.nullHeuristic)
        print "Path result for A* with letter heuristic:",search.aStarSearch(self,letterHeuristic)


"""THIS CODE YOU CAN CHANGE"""

def letterHeuristic(position, problem):
    return ord(position)-ord(problem.goal)

"""agraph is a graph with edges AB, AD, BC and CD, goal node D, start node A,  and appropriate node costs."""
agraph = TestVerySimpleGraph([('A','B',1),('A','D',10),('B','C',1),('C','D',1)],'D','A')
agraph.runTest()
