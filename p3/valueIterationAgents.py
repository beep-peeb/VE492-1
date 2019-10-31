import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #print(type(self.iterations))
        for i in range(self.iterations):
            counter = util.Counter()
            states = self.mdp.getStates()
            for state in states:
                #print(state)
                actions = self.mdp.getPossibleActions(state)
                actionValue = float("-inf") # get the maximum action, initialized to be the smallest
                for action in actions:
                    #print(self.computeQValueFromValues(state,action))
                    if self.computeQValueFromValues(state,action) > actionValue:
                        actionValue = self.computeQValueFromValues(state,action)
                    counter[state] = actionValue

            self.values = counter







    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        q_value = 0
        q_function = self.mdp.getTransitionStatesAndProbs(state,action)
        for state_1,prob in q_function:
            q_value = q_value + prob * (self.mdp.getReward(state, action, state_1) + self.discount * self.values[state_1])

        return q_value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)

        if len(actions) == 0:
            return None

        values = util.Counter()
        for action in actions:
            values[action] = self.computeQValueFromValues(state,action)

        return values.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        num = len(states)
        for i in range(self.iterations):
            state = states[i % num]
            value = []
            if self.mdp.isTerminal(state) == False:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    value.append(self.computeQValueFromValues(state,action))
                self.values[state] = max(value)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        priorityqueue = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = {} # in the set to avoid duplicate
        for state in states:
            if self.mdp.isTerminal(state) == False:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    for state_1,prob in self.mdp.getTransitionStatesAndProbs(state,action):
                        if state_1 in predecessors:
                            predecessors[state_1].add(state)
                        else:
                            predecessors[state_1] = {state}

        for state in states:
            if self.mdp.isTerminal(state) == False:
                actions = self.mdp.getPossibleActions(state)
                q_value = float("-inf")
                for action in actions:
                    temp = self.computeQValueFromValues(state,action)
                    if temp > q_value:
                        q_value = temp #get the maximum value
                diff = abs(self.values[state] - q_value)
                priorityqueue.update(state,-diff)

        for i in range(self.iterations):
            if priorityqueue.isEmpty():
                break #terminate
            state = priorityqueue.pop()
            if self.mdp.isTerminal(state) == False:
                actions = self.mdp.getPossibleActions(state)
                value = float("-inf")
                for action in actions:
                    temp_value = self.computeQValueFromValues(state,action)
                    if temp_value > value:
                        value = temp_value
                self.values[state] = value

            for pre in predecessors[state]:
                if self.mdp.isTerminal(pre) == False:
                    actions = self.mdp.getPossibleActions(pre)
                    q_values = float("-inf")
                    for action in actions:
                        temps = self.computeQValueFromValues(pre,action)
                        if temps >= q_values:
                            q_values = temps
                    diff = abs(q_values - self.values[pre])

                    if diff > self.theta:
                        priorityqueue.update(pre,-diff)






