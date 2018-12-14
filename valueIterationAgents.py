# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

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
       
        # Hoe weet je hoe groot je wereld is?
        # Is er al een manier om je buurstates te zien of moet dat zelf worden geimplementeerd?
        

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
<<<<<<< HEAD
                
    
=======

        for y in mdp.getStates():
            self.values[y] = 0

        for x in range(iterations):
            for y in mdp.getStates():
                pa = mdp.getPossibleActions(y)
                for z in pa:
                    self.computeQValueFromValues(y, z) 
                #print 99
                #print self.values[y]
                self.updateValue(y) 
                #print self.values[y]            

                    # tsap = mdp.getTransitionStatesAndProbs(y, z)
                    # totval = 0
                    # for a in tsap
                    #     i j = a
                    #     totval += j * getValue(i)
        for y in mdp.getStates():
            self.getPolicy(y)                



>>>>>>> 9eee2b8015b43dd6e7ba32edb2c4725c7e4104ae
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
        # StatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        # totval = 0
        # for nextState, probability in StatesAndProbs:
        #     (x, y) = a
        #     #print self.discount 
        #     print 'jaflajdf'
        #     if(self.mdp.isTerminal(x)):
        #         print 'Yoeyou'
        #     print self.getValue(x)
        #     print 'Reward:'
        #     print self.mdp.getReward(state, action, x)
        #     totval += y * (self.mdp.getReward(state, action, x) + self.discount * self.getValue(x))
        #     print totval
        value = 0
        transitionFunction = self.mdp.getTransitionStatesAndProbs(state,action)
        for nextState, probability in transitionFunction:
            value += probability * (self.mdp.getReward(state, action, nextState) 
                  + (self.discount * self.values[nextState]))

        #return value
        print value
        self.values[(state, action)] = value

    def updateValue(self, state):
        if (not self.mdp.isTerminal(state)):
            directionlist = ['North', 'East', 'South', 'West']
            maxval = None
            for x in directionlist:
                if maxval is None or self.values[(state, x)] > maxval:
                    maxval = self.values[(state, x)]
            self.values[state] = maxval
        else:
            self.values[state] = self.mdp.getReward(state,(),state)
            print self.mdp.getReward(state,(),state)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
<<<<<<< HEAD
        return []
       
=======
        directionlist = ['North', 'East', 'South', 'West']
        val = self.values[state]
        action = 'North'
        for x in directionlist:
            if (self.values[(state, x)] == val):
                return action
>>>>>>> 9eee2b8015b43dd6e7ba32edb2c4725c7e4104ae

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
