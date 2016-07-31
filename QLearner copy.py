"""
Template for implementing QLearner  (c) 2015 Tucker Balch
CS 7646 project 5
QLnearner implementation
By Zhongtao Yang
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.allActions = range(self.num_actions)
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        # initialize the Q Table
        self.Q = np.zeros((self.num_states,self.num_actions),dtype=float)
        # initialize the R,T Table
        # self.R = self.Q
        
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        prevState = self.s
        prevAction = self.a
        newState = s_prime
        # decide what action to take in the new state
        if np.random.random() < self.rar: #Random exploration
            action = rand.randint(0, self.num_actions-1)
        else: # Or choose the action best on Q table at current state
            possibleAction = self.Q[newState]
            maxQVal = np.max(possibleAction)
            action = np.where(possibleAction==maxQVal)[0][0]     
        prevQ = self.Q[prevState][prevAction]
        newQ = self.Q[newState][action]
        # update the Q based on new state and new action made from prev state
        self.Q[prevState][prevAction] = (1.0 - self.alpha) * prevQ + self.alpha * (self.gamma * newQ + r)
        self.s = s_prime
        self.a = action
        # decay the rar, otherwise it will always try explore randomly with high possibility
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
