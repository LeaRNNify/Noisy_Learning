import time

import numpy as np
from dfa import DFA


class CounterDFA():
    def __init__(self, init_state, final_states, transitions, alphabet2counter,
                 init_tokens, sup=True):
        """ 
        Works like a DFA, where all the transitions also remove/add
        tokens from/to a single counter.
        A word w is in the language if:
            1. counter >= 0 and w reaches an accepting state.
            2. counter < 0 and sup = True
        :param 
        alphabet2counter: mapping from alphabet to an integer.
        init_tokens: the initial amount of tokens
        sup: if true the the language of the counter DFA is a sup
             of the dfa language.
        
        """
        self.init_state = init_state
        self.final_states = final_states
        self.transitions = transitions
        self.init_tokens = init_tokens
        self.alphabet = list(transitions[init_state].keys())
        self.sup = sup
        self.alphabet2counter = alphabet2counter

    def is_word_in(self, word):
        counter = self.init_tokens
        state = self.init_state
        for letter in word:
            state = self.transitions[state][letter]
            counter += self.alphabet2counter[letter]
        if counter < 0:
            return self.sup
        else:
            return state in self.final_states

    def __repr__(self):
        return "initial tokens = {} \n" \
               "is sup: {} \n" \
               "alphabet to counter: {}".format(self.init_tokens, self.sup, self.alphabet2counter)


def from_dfa_to_rand_counter_dfa(dfa: DFA) -> CounterDFA:
    alphabet = dfa.alphabet
    init_tokens = np.random.randint(0, len(alphabet))

    if np.random.randint(1, 3) == 2:
        sup = True
    else:
        sup = False

    alphabet2counter = {}
    while -1 not in alphabet2counter.values():
        alphabet2counter = {}
        for l in alphabet:
            if np.random.randint(0, len(alphabet)) <= len(alphabet) / 4:
                alphabet2counter.update({l: -1})
            else:
                counter = np.random.randint(1, 6)
                alphabet2counter.update({l: counter})

    return CounterDFA(dfa.init_state, dfa.final_states, dfa.transitions, alphabet2counter,
                      init_tokens, sup)


class NoisyCounterDFA:
    def __init__(self, init_state, final_states, transitions, alphabet2counter,
                 init_tokens, noise_prob, sup=True):
        """
        Works like a DFA, where all the transitions also remove/add
        tokens from/to a single counter.
        A word w is in the language if:
            1. counter >= 0 and w reaches an accepting state.
            2. counter < 0 and sup = True
        :param
        alphabet2counter: mapping from alphabet to an integer.
        init_tokens: the initial amount of tokens
        sup: if true the the language of the counter DFA is a sup
             of the dfa language.

        """
        self.init_state = init_state
        self.final_states = final_states
        self.transitions = transitions
        self.init_tokens = init_tokens
        self.alphabet = list(transitions[init_state].keys())
        self.sup = sup
        self.alphabet2counter = alphabet2counter
        self.noise_prob = noise_prob
        self.known_words = {}
        self.knows = []

    def is_word_in(self, word):
        prev_ans = self.known_words.get(word, None)
        if prev_ans is not None:
            return prev_ans

        counter = self.init_tokens
        state = self.init_state
        for letter in word:
            state = self.transitions[state][letter]
            counter += self.alphabet2counter[letter]

        if counter < 0:
            label = self.sup & (np.random.randint(0, int(1 / self.noise_prob)) != 0)
            # if np.random.randint(0, int(1 / self.noise_prob)) == 0:
            #     label = not self.sup
            # else:
            #     label = self.sup
        else:
            label = state in self.final_states
        self.known_words.update({word: label})

        return label

    def __repr__(self):
        return "noisy counter DFA \n" \
               "noise_prob = {} \n" \
               "initial tokens = {} \n" \
               "is sup = {} \n" \
               "alphabet to counter = {}".format(self.noise_prob, self.init_tokens, self.sup, self.alphabet2counter)


def from_counter_to_noisy_counter(counter_dfa, noise_prob=0.01):
    return NoisyCounterDFA(counter_dfa.init_state,
                           counter_dfa.final_states,
                           counter_dfa.transitions,
                           counter_dfa.alphabet2counter,
                           counter_dfa.init_tokens,
                           noise_prob,
                           counter_dfa.sup)


# def from_dfa_to_rand_counter_dfa(dfa: DFA) -> CounterDFA:
#     alphabet = dfa.alphabet
#     init_tokens = np.random.randint(0, len(alphabet))
#
#     if np.random.randint(1, 3) == 2:
#         sup = True
#     else:
#         sup = False
#
#     alphabet2counter = {}
#     while -1 not in alphabet2counter.values():
#         alphabet2counter = {}
#         for l in alphabet:
#             if np.random.randint(0, len(alphabet)) <= len(alphabet) / 4:
#                 alphabet2counter.update({l: -1})
#             else:
#                 counter = np.random.randint(1, 6)
#                 alphabet2counter.update({l: counter})
#
#     return CounterDFA(dfa.init_state, dfa.final_states, dfa.transitions, alphabet2counter,
#                       init_tokens, sup)


class DFAFinalCount:
    def __init__(self, init_state, final_states, transitions, multiplier, threshold, second_add=0):
        self.threshold = threshold
        self.final_states = final_states
        self.init_state = init_state
        self.transitions = transitions
        self.states = list(transitions.keys())
        self.alphabet = list(transitions[init_state].keys())
        self.current_state = self.init_state
        self.multiplier = multiplier
        self.second_add = second_add
        self.states_close_to_final = []
        for s in self.states:
            if s in final_states:
                continue
            for l in self.alphabet:
                if self.next_state_by_letter(s, l) in final_states:
                    self.states_close_to_final.append(s)
                    break
        # print(self.final_states)
        # print(self.states_close_to_final)
        # print(self.states)
        # # time.sleep(10)
        # for state in self.states:
        #     if (state not in self.final_states) and (state not in self.states_close_to_final):
        #         print(state)
        #         break

        self.init_count = float(self.init_state in self.final_states) + float(self.second_add * (self.init_state in self.states_close_to_final))

        # print(len(final_states))
        # print(len(self.states_close_to_final))

    def is_word_in(self, word):
        state = self.init_state
        count = self.init_count
        # if len(word) == 0:
        #     return init_state in final_states
        for letter in word:
            count *= self.multiplier
            state = self.transitions[state][letter]
            count += (state in self.final_states) + self.second_add * (state in self.states_close_to_final)
            # if (state not in self.final_states) and (state not in self.states_close_to_final):
            #     print("here")
            #     time.sleep(100)
            # print(count)
        # if (state in self.final_states) and (count <= self.threshold):
        #     print(word)
        #     time.sleep(1)
        return count > self.threshold

    def next_state_by_letter(self, state, letter):
        next_state = self.transitions[state][letter]
        return next_state

    def is_final_state(self, state):
        return state in self.final_states

    def is_word_letter_by_letter(self, letter, reset=False):
        if reset:
            self.current_state = self.init_state

        self.current_state = self.next_state_by_letter(self.current_state, letter)
        return self.current_state in self.final_states

    def reset_current_to_init(self):
        self.current_state = self.init_state


def from_dfa_to_dfa_final_count(dfa: DFA, multiplier: float, threshold: float, second_add: float) -> DFAFinalCount:
    return DFAFinalCount(dfa.init_state, dfa.final_states, dfa.transitions, multiplier, threshold, second_add)
