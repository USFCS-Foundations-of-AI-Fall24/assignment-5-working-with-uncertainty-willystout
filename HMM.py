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
                emit_str = line.rstrip('\n').split(" ")
                if len(emit_str) == 3:
                    state, emit, prob = emit_str
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    self.emissions[state][emit.lower()] = prob

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
    # """return an n-length Sequence by randomly sampling from this HMM."""
    ## I had chatGPT help with explaining how to use numpy.random.choice, and how to use
    ## the transition probabilities to determine the next state.
    def generate(self, n):
        state = '#'  # initial state
        stateseq = []  # state seq
        emitseq = []  # emit seq

        for i in range(n):
            next_states = list(
                self.transitions[state].keys())  # get the next transition states based on the current state
            state_probs = [float(self.transitions[state][next_state]) for next_state in
                           next_states]  # get an array of probabilities in order to choose the next states
            state = numpy.random.choice(next_states,
                                        p=state_probs)  # choose the next states based on the transition probability
            stateseq.append(state)  # add that chosen state to the state sequence

            emissions = list(self.emissions[state].keys())  # get the emit based on the current state
            emission_probs = [float(self.emissions[state][emit]) for emit in
                              emissions]  # get an array of probabilities in order to choose the next emit
            emit = numpy.random.choice(emissions,
                                       p=emission_probs)  # choose the next emission based on each emit probability
            emitseq.append(emit)  # add the chosen emit to the emit sequence

        return Sequence(stateseq, emitseq)

    # used chatGPT to help me understand how to implement the forward algorithm based on pseudocode provided in lecture
    def forward(self, sequence):
        ## set up the initial maxtrix M, with P=1 for the # state. Use a 2d array
        # for each state on day 1: P(state given emit) = alpha P(emit given state) * P(state given #)
        # for i = 2 to n
        #   for each state s
        #       sum = 0
        #       for s2 in states:
        #          sum += M[s2, i-1] * T[s2, s] * E[emit[i], s]
        #       M[s, i] = sum
        n = len(sequence)
        states = list(self.emissions.keys())  # [happy, grumpy, hungry] for cat example
        m = {state: [0] * n for state in states}  # Initialize matrix M with zeros for each state
        first_emit = sequence.outputseq[0]
        for state in states:
            init_prob = float(self.transitions['#'][state])  # P(state|#)
            emit_prob = float(self.emissions[state][first_emit])  # P(emit|state)
            m[state][0] = init_prob * emit_prob

        for t in range(1, n):
            emit = sequence.outputseq[t]  # get the emit for the
            for curr_state in states:  # for each state
                prob_sum = 0
                for prev_state in states:  # for each previous state
                    transition_prob = float(self.transitions[prev_state][curr_state])  # get trans prob
                    emission_prob = float(self.emissions[curr_state][emit])  # get emit prob
                    prob_sum += m[prev_state][
                                    t - 1] * transition_prob * emission_prob  # add the sum of prev state * trans prob * emit prob
                m[curr_state][t] = prob_sum  # set matrix at curr state and row t to the prob sum
        print(m)
        return sum(m[state][n - 1] for state in states)  # return the sum of the last row of the matrix

    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.

    def viterbi(self, sequence):
        # Used Claude to explain the Viterbi algorithm and how to implement it based on the pseudocode provided in lecture
        """
            Implementation of the Viterbi algorithm.

            Parameters:
            observations: List of observations
            states: List of possible hidden states
            start_prob: Dictionary of initial state probabilities
            trans_prob: Dictionary of transition probabilities between states
            emit_prob: Dictionary of emission probabilities for each state

            Returns:
            most_likely_path: List of most likely sequence of hidden states
        """
        ## initialization of the algorithm
        n = len(sequence)
        states = list(self.emissions.keys())  # [happy, grumpy, hungry] for cat example
        viterbi_prob = {state: [0] * n for state in states}  # Initialize matrix M with zeros for each state
        backpointer = {state: [None] * n for state in states}  # initalize backpointer matrix to be empty
        first_emit = sequence.outputseq[0].lower()
        for state in states:
            if first_emit in self.emissions[state]:
                emit_prob = float(self.emissions[state][first_emit])
            else:
                emit_prob = "1e-10"
            initial_prob = float(self.transitions['#'][state]) * float(emit_prob)
            viterbi_prob[state][0] = initial_prob
            backpointer[state][0] = 0

        ## Main loop of the algorithm
        for t in range(1, n):
            curr_emit = sequence.outputseq[t].lower()
            for curr_state in states:
                max_prob = 0
                best_prev_state = None
                for prev_state in states:
                    if curr_emit in self.emissions[curr_state]:
                        emit_prob = self.emissions[curr_state][curr_emit]
                    else:
                        emit_prob = "1e-10"
                    trans_prob = (
                            (float(viterbi_prob[prev_state][t - 1]) *
                             float(self.transitions[prev_state][curr_state])) *
                            float(emit_prob)
                    )
                    if trans_prob > max_prob:
                        max_prob = trans_prob
                        best_prev_state = prev_state
                viterbi_prob[curr_state][t] = max_prob
                backpointer[curr_state][t] = best_prev_state
        ## final backtracking step in algorithm
        final_t = n - 1
        best_final_state = None
        max_final_prob = 0
        for state in states:
            if viterbi_prob[state][final_t] > max_final_prob:
                max_final_prob = viterbi_prob[state][final_t]
                best_final_state = state

        result = [best_final_state]
        curr_state = best_final_state

        for t in range(final_t, 0, -1):
            prev_state = backpointer[curr_state][t]
            result.insert(0, prev_state)
            curr_state = prev_state
        return result

    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basename', type=str, help='basename for HMM')
    parser.add_argument('--generate', type=int, help='number of sequences to generate')
    parser.add_argument('--forward', action='store_true', help='run forward algorithm')
    parser.add_argument('--viterbi', type=str, help='observation sequence file to run viterbi on')
    args = parser.parse_args()
    hmm = HMM()
    hmm.load(args.basename)
    if args.generate:
        newSeq = hmm.generate(args.generate)
    else:
        newSeq = hmm.generate(10)
    if args.forward:
        print(hmm.forward(newSeq))
    ## used claude to help with Viterbi command line argument
    if args.viterbi:
        with open(args.viterbi) as f:
            for line in f:
                obs = line.strip().split()
                if obs:
                    # Convert to uppercase to match emission format
                    seq = Sequence([], obs)
                    print(hmm.viterbi(seq))
