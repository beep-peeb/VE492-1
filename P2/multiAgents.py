from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent
import search
from search import bfs
from search import astar


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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        """for ghost in newGhostStates:
            print(ghost.getPosition())"""

        if action == "Stop":
            return -50

        newscore = successorGameState.getScore()
        distance1 = manhattanDistance(newPos, newGhostStates[0].getPosition())

        for scaretime in newScaredTimes:
            newscore = newscore + scaretime / (distance1 + 1)

        # bonus for being scared

        if distance1 < 5 and distance1 > 0:
            newscore = newscore - 5/distance1

        # the longer distance, the smaller effect

        foodlist = newFood.asList()

        distancelist = []
        for food in foodlist:
            distancelist.append(manhattanDistance(newPos, food))

        if len(foodlist) > 0:
            newscore = newscore + 5 / min(distancelist)

        if newPos in foodlist:
            newscore = 1.1 * newscore
        return newscore


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        ghostNum = gameState.getNumAgents() - 1
        return self.maxvalue(gameState, 1, ghostNum)

        util.raiseNotDefined()

    def maxvalue(self, gameState, depth, GhostNum):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        maxv = float("-inf")
        action = Directions.STOP
        for actions in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, actions)
            minv = self.minvalue(nextState, depth, GhostNum, 1)

            if minv > maxv:
                maxv = minv
                action = actions

        if depth > 1:
            return maxv
        return action

    def minvalue(self, gameState, depth, GhostNum, index):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        minv = float("inf")

        ActionSet = gameState.getLegalActions(index)
        nextState = []
        for action in ActionSet:
            nextState.append(gameState.generateSuccessor(index, action))

        if index != GhostNum:
            for next in nextState:
                minv = min(minv, self.minvalue(next, depth, GhostNum, index + 1))

        else:
            if depth != self.depth:
                for next in nextState:
                    minv = min(minv, self.maxvalue(next, depth + 1, GhostNum))
            else:
                for next in nextState:
                    minv = min(minv, self.evaluationFunction(next))

        return minv


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        ghostNum = gameState.getNumAgents() - 1

        return self.maxval(gameState,1,ghostNum,float("-inf"),float("inf"))
        util.raiseNotDefined()

    def maxval(self,gameState,depth,ghostNum,alpha,beta):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v = float("-inf")
        action = Directions.STOP

        for actions in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,actions)
            minv = self.minval(nextState,depth,ghostNum,1,alpha,beta)

            if minv > v:
                v = minv
                action = actions

            if v > beta:
                #why not >=
                #on the slide it should be >= however in the test case would show wrong answer
                return v

            alpha = max(alpha,v)

        if depth > 1:
            return v
        return action



    def minval(self,gameState,depth,ghostNum,index,alpha,beta):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v = float("inf")
        ActionSet = gameState.getLegalActions(index)
        nextState = []
        for action in ActionSet:
            next = gameState.generateSuccessor(index,action)

            if index!= ghostNum:
                minv = self.minval(next,depth,ghostNum,index+1,alpha,beta)
            else:
                if depth!=self.depth:
                    minv = self.maxval(next,depth+1,ghostNum,alpha,beta)
                else:
                    minv = self.evaluationFunction(next)

            v = min(v,minv)

            if v < alpha:
                #why not <=
                return v
            beta = min(beta,v)
        return v




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

        ghostNum = gameState.getNumAgents() - 1
        return self.maxval(gameState,1,ghostNum)

        util.raiseNotDefined()

    def maxval(self,gameState,depth,ghostNum):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        maxv = float("-inf")
        action = Directions.STOP
        for actions in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, actions)
            expectimaxvalue = self.expectimaximum(nextState, depth, ghostNum, 1)


            if expectimaxvalue > maxv:
                maxv = expectimaxvalue
                action = actions

        if depth > 1:
            return maxv
        return action

    def expectimaximum(self,gameState,depth,ghostNum,index):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        ActionSet = gameState.getLegalActions(index)

        value = 0
        actionnum = len(ActionSet)
        chance = 1.0/actionnum
        nextState = []
        for action in ActionSet:
            nextState.append(gameState.generateSuccessor(index,action))

        for next in nextState:
            #first define the state then define the value, bug fixed
            if index!=ghostNum:
                value = value + chance*self.expectimaximum(next,depth,ghostNum,index+1)
            else:
                if depth!=self.depth:
                    value = value + chance*self.maxval(next,depth+1,ghostNum)
                else:
                    value = value + chance*self.evaluationFunction(next)
        return value



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    from search import bfs

    #new information
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    
    #food storing
    foodlist = []
    food = newFood.asList()
    for eatfood in food:
        foodlist.append(mazeDistance(eatfood,newPos,currentGameState))

    # the effect of ghost
    for ghost in newGhostStates:
        #print(ghost)
        if mazeDistance(newPos,ghost.getPosition(),currentGameState) > 0:
            if len(foodlist)> 0:
            #eating the closet food
                if mazeDistance(newPos,ghost.getPosition(),currentGameState)==0:
                    score = -score
                else:
                    score = score + 10/min(foodlist)
            else:
                if ghost.scaredTimer > 0:
                    #scared ghosts
                    score = score + 5/mazeDistance(newPos,ghost.getPosition(),currentGameState)
                else:
                    score = score - 10/mazeDistance(newPos,ghost.getPosition(),currentGameState)

    return score

    util.raiseNotDefined()

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=(x1,y1), goal=(x2,y2), warn=False, visualize=False)
    return len(search.astar(prob,manhattanHeuristic))

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

# Abbreviation
better = betterEvaluationFunction

