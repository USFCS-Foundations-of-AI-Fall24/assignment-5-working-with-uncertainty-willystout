import random
import argparse
import codecs
import os
from idlelib.pyparse import trans

import numpy


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# HMM model
class HMM:
    def __init__(self, transitions=None, emissions=None):
        if emissions is None:
            emissions = {}
        if transitions is None:
            transitions = {}
        self.transitions = transitions
        self.emissions = emissions
        # """creates a model from transition and emission probabilities
        # e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
        #       'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
        #       'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""

    ## part 1 - you do this.
    def load(self, basename):
        with open(basename + ".trans") as f:
            for line in f:
                trans_str = line.strip().split(" ")
                if len(trans_str) == 3:
                    start, finish, prob = trans_str
                    if start not in self.transitions:
                        self.transitions[start] = {}
                    self.transitions[start][finish] = prob
        with open(basename + ".emit") as f:
            for line in f:
                emit_str = line.strip().split(" ")
                if len(emit_str) == 3:
                    state, emit, prob = emit_str
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    self.emissions[state][emit] = prob

    ## you do this.
    # One cool thing we can do with an HMM is Monte Carlo simulation.
    # We'll do this using generate. So implement that next. It should
    # take an integer n, and return a Sequence of length n. To generate
    # this, start in the initial state and repeatedly select successor
    # states at random, using the transition probability as a weight,
    # and then select an emission, using the emission probability as a
    # weight. You may find either numpy.random.choice or random.choices
    # very helpful here. Be sure that you are using the transition
    # probabilities to determine the next state, and not a uniform
    # distribution!
     #"""return an n-length Sequence by randomly sampling from this HMM."""
    ## I had chatGPT help with explaining how to use numpy.random.choice, and how to use
    ## the transition probabilities to determine the next state.
    def generate(self, n):
        state = '#' # initial state
        stateseq = [] # state seq
        emitseq = [] # emit seq

        for i in range(n):
            next_states = list(self.transitions[state].keys()) # get the next transition states based on the current state
            state_probs = [float(self.transitions[state][next_state]) for next_state in next_states] # get an array of probabilities in order to choose the next states
            state = numpy.random.choice(next_states, p=state_probs) # choose the next states based on the transition probability
            stateseq.append(state) # add that chosen state to the state sequence

            emissions = list(self.emissions[state].keys()) # get the emit based on the current state
            emission_probs = [float(self.emissions[state][emit]) for emit in emissions] # get an array of probabilities in order to choose the next emit
            emit = numpy.random.choice(emissions, p=emission_probs) # choose the next emission based on each emit probability
            emitseq.append(emit) # add the chosen emit to the emit sequence

        return Sequence(stateseq, emitseq)



    def forward(self, sequence):
        pass


    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.


    def viterbi(self, sequence):
        pass


    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basename', type=str, help='basename for HMM')
    parser.add_argument('--generate', type=int, help='number of sequences to generate')
    args = parser.parse_args()
    hmm = HMM()
    hmm.load(args.basename)
    print(hmm.generate(args.generate))
