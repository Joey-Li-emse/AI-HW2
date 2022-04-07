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


from mimetypes import init
from pacman import SCARED_TIME
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
        init_pos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # Position of ghost do not change regardless of your state 
        # because you can't predict the future
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        # Count down from 40 moves
        ghostStartPos = [ghostState.start.getPosition() for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        # If the next move bring to food +10 
        def food_next(next_PAC, newFood, score):
          if newFood[next_PAC[0]][next_PAC[1]] :
            return score + 50
          return score 
        
        # If the next move brings us closer to the closest food +10 
        def closest_food(food, currentGS, nextGS, score):
          init_PAC = currentGS.getPacmanPosition()
          next_PAC = nextGS.getPacmanPosition()
          mini = float("inf")
          coord = None
          for i in range(food.width) :
            for j in range(food.height):
              if food[i][j] :
                tmp = (i, j)
                if manhattanDistance(tmp, init_PAC) < mini : 
                  mini = manhattanDistance(tmp, init_PAC)
                  coord = tmp
          if coord :
            if manhattanDistance(next_PAC, coord) < mini: 
              score += 30
        
          return score

        # If pacman get closer to the ghost (we will pass the closest ghost as arguemnt) it get malus 
        def closer_to(element, currentGameState, next_PAC):
          init_PAC = currentGameState.getPacmanPosition()
          init_distance = 0.0
          next_distance = 0.0
          
          init_distance += manhattanDistance(init_PAC, element)
          next_distance += manhattanDistance(next_PAC, element)

          if init_distance > next_distance :
            return True 
          return False

        #if it brings closer to ghost, malus 
        def closer_to_ghost(element, currentGameState, next_PAC, score):
          if closer_to(element, currentGameState, next_PAC):
            score -= 10
          return score

        # Return the Closes ghost to pacman
        def closest_ghost(next_Pac, ghostPositions):
          min = float("inf")
          for ghost in ghostPositions:
            if manhattanDistance(next_Pac, ghost) < min :
              min = manhattanDistance(next_Pac, ghost)
              closest = ghost 
          return closest, min  

        # Generate the forbidden positons 
        def generate_forbidden_zone(ghostPositions):
          forbidden_zone = []
          for ghost in ghostPositions:
            north = (ghost[0], ghost[1] + 1)
            south = (ghost[0], ghost[1] - 1)
            east = (ghost[0] + 1, ghost[1])
            west = (ghost[0] - 1, ghost[1])

            forbidden_zone.append(north)
            forbidden_zone.append(south)
            forbidden_zone.append(east)
            forbidden_zone.append(west)
            
          # print(forbidden_zone)
          # a = raw_input()
          return forbidden_zone
        
        # Forbid pacman to go to the a next possible ghost position 
        def forbidden(next_Pac, score):
          if next_Pac in generate_forbidden_zone(ghostPositions):
            score -= 500000
          return score 

        # Return the closest capsule
        def Closest_Capsule(Capsules, init_pos):
          if Capsules:
            dist = float("inf") 
            for Capsule in Capsules: 
              if manhattanDistance(Capsule, init_pos) < dist :
                dist = manhattanDistance(Capsule, init_pos)
                closest_Caps = Capsule
            return Capsule, dist
          return None
        
        # if ghost close : go to capsule if there is capsule left 
        def get_Capsule(Capsules, food, scared_time, score):
          if Capsules :
            
            if scared_time[0]: 
              
              ghost, dist = closest_ghost(newPos, ghostPositions)
              area = food.width * food.height        
              safety = area**(1/2.)/2.
              if dist < safety :
                Caps = Closest_Capsule(Capsules, init_pos)[0]
                if Caps or dist < 4:
                  if closer_to(Caps, currentGameState, newPos):
                    score +=30 
          return score

        # If in Godmode, chase after the ghost 
        def FCK_THE_GHOST(scared, currentGS, next_PAC, score):
          if scared [0]:
            VIKTIM = closest_ghost(next_PAC, ghostPositions)[0]
            if closer_to(VIKTIM, currentGS, next_PAC):
              score += 40
              if next_PAC in generate_forbidden_zone(ghostPositions):
                score += 1000000
          return score

      


        score = 0    
        generate_forbidden_zone(ghostPositions)
        score = forbidden(newPos, score) 
        score = closer_to_ghost(closest_ghost(newPos, ghostPositions)[0], currentGameState, newPos, score) #if pacman is going cloaser to the closest ghost it's bad 
        score = food_next(newPos, newFood, score)  #if it can eat food on it's next move, it's good 
        score = closest_food(newFood, currentGameState, successorGameState, score) #if goes closer to the closest food, that's good 
        score = get_Capsule(newCapsule, newFood, newScaredTimes, score)
        score = FCK_THE_GHOST(newScaredTimes, currentGameState, newPos, score)
        
        print(action)
        print(score)
        a = raw_input()
        return  score 
        #please change the return score as the score you want

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
        self.current_agent = 0
        

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
        """
        "*** YOUR CODE HERE ***"

        return self.minimax_decision(gameState, 0, 0)

    def minimax_decision(self, gameState, agent_id, depth):

      if depth == self.depth or gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)

      if agent_id == 0: 
        return self.maxplay(gameState, agent_id, depth)
      
      if agent_id > 0: 
        return self.minplay(gameState, agent_id, depth) 



    def maxplay(self, gameState, agent_id, depth):
      
      next_agent = agent_id + 1 
      best_action = Directions.STOP 
      best_score = - float("inf")
      actions = gameState.getLegalActions(agent_id)

      for action in actions:
        score = self.minimax_decision(gameState.generateSuccessor(agent_id, action), next_agent, depth) 
        if score > best_score:
          best_action = action
          best_score = score
      if depth == 0 :
        return best_action

      return best_score


    def minplay(self, gameState, agent_id, depth):
      
      next_agent = agent_id + 1
      next_depth = depth
      
      if next_agent == gameState.getNumAgents():
        next_depth += 1  
        next_agent = 0
       
      best_score = float("inf")
      actions = gameState.getLegalActions(agent_id)
      
      for action in actions:
        score = self.minimax_decision(gameState.generateSuccessor(agent_id, action), next_agent, next_depth)
        if score < best_score:
          best_action = action
          best_score = score 

      return best_score
    
    
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        max = float("inf")
        min = -float("inf")
        return self.ABP_decision(gameState, 0, 0, min, max)
        util.raiseNotDefined()

    def ABP_decision(self, gameState, agent_id, depth, alpha, beta):

      if depth == self.depth or gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)

      if agent_id == 0: 
        return self.maxplay(gameState, agent_id, depth, alpha, beta)
      
      if agent_id > 0: 
        return self.minplay(gameState, agent_id, depth, alpha, beta) 



    def maxplay(self, gameState, agent_id, depth, alpha, beta):
      
      next_agent = agent_id + 1 
      best_action = Directions.STOP 
      best_score = - float("inf")
      actions = gameState.getLegalActions(agent_id)

      for action in actions:
        score = self.ABP_decision(gameState.generateSuccessor(agent_id, action), next_agent, depth, alpha, beta) 
        if score > best_score:
          best_action = action
          best_score = score
        alpha = max(alpha, best_score)
        if beta < alpha: 
          break
      if depth == 0 :
        return best_action

      return best_score


    def minplay(self, gameState, agent_id, depth, alpha, beta):
      
      next_agent = agent_id + 1
      next_depth = depth
      
      if next_agent == gameState.getNumAgents():
        next_depth += 1  
        next_agent = 0
       
      best_score = float("inf")
      actions = gameState.getLegalActions(agent_id)
      
      for action in actions:
        score = self.ABP_decision(gameState.generateSuccessor(agent_id, action), next_agent, next_depth, alpha, beta)
        if score < best_score:
          best_action = action
          best_score = score 
        beta = min(beta, best_score)
        if beta < alpha: 
          break
      return best_score
    
    

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
        return self.Expectimax_decision(gameState, 0, 0)
        util.raiseNotDefined()
        

    def Expectimax_decision(self, gameState, agent_id, depth):

      if depth == self.depth or gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)

      if agent_id == 0: 
        return self.maxplay(gameState, agent_id, depth)
      
      if agent_id > 0: 
        return self.randplay(gameState, agent_id, depth) 



    def maxplay(self, gameState, agent_id, depth):
      
      next_agent = agent_id + 1 
      best_action = Directions.STOP 
      best_score = - float("inf")
      actions = gameState.getLegalActions(agent_id)

      for action in actions:
        score = self.Expectimax_decision(gameState.generateSuccessor(agent_id, action), next_agent, depth) 
        if score > best_score:
          best_action = action
          best_score = score
      if depth == 0 :
        return best_action

      return best_score


    def randplay(self, gameState, agent_id, depth):
      
      next_agent = agent_id + 1
      next_depth = depth
      
      if next_agent == gameState.getNumAgents():
        next_depth += 1  
        next_agent = 0
       
      
      actions = gameState.getLegalActions(agent_id)
      
      score = 0 
      for action in actions:
        prob = 1./len(actions)
        score += prob *self.Expectimax_decision(gameState.generateSuccessor(agent_id, action), next_agent, next_depth)

      return score
    
    
        
        


