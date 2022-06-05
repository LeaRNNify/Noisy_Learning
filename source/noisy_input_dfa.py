from time import sleep

import numpy as np
from dfa import DFA
from randwords import scrumble_word, scrumble_word_orderly, is_words_in_dfa, scrumble_word_reducing


class NoisyInputDFA:
    def __init__(self, init_state, final_states, transitions, mistake_prob=0.001):
        """

        """
        self.init_state = init_state
        self.final_states = final_states
        self.transitions = transitions
        self.mistake_prob = mistake_prob
        self.alphabet = list(transitions[init_state].keys())
        self.known_mistakes = {}
        self.scrumble_prob = 0.01
        self.scrumble_prob_inc = 1.5

    def is_word_in(self, word):
        prev_ans = self.known_mistakes.get(word, None)
        if prev_ans is not None:
            return prev_ans

        new_word = tuple(scrumble_word(word, self.alphabet, self.mistake_prob))
        # new_word = list(word)
        # for i in range(len(new_word)):
        #     if np.random.randint(0, int(1 / self.mistake_prob)) == 0:
        #         new_word[i] = self.alphabet[np.random.randint(0, int(len(self.alphabet)))]

        state = self.init_state
        for letter in new_word:
            state = self.transitions[state][letter]
        label = state in self.final_states
        self.known_mistakes.update({word: label})
        return label

    #
    # def is_word_in(self, word):
    #     prev_ans = self.known_mistakes.get(word, None)
    #     if prev_ans is not None:
    #         return prev_ans
    #
    #     new_word = word
    #     state = self.init_state
    #     for letter in new_word:
    #         state = self.transitions[state][letter]
    #     label_out = state in self.final_states
    #
    #     if np.random.randint(0, int(1 / self.mistake_prob)) == 0:
    #         self.known_mistakes.update({new_word: not label_out})
    #         new_word = []
    #         for l in word:
    #             new_word.append(l)
    #             if tuple(new_word) not in self.known_mistakes:
    #                 self.update_known(tuple(new_word))
    #
    #         # print("before len")
    #         # print(len(self.known_mistakes))
    #         # print("before:{}".format(len(self.known_mistakes)))
    #         for a1 in self.alphabet:
    #             new_word = list(word)
    #             new_word.pop(-1)
    #             new_word.pop(-1)
    #             new_word[-1] = a1
    #             if tuple(new_word) not in self.known_mistakes:
    #                 self.update_known(tuple(new_word))
    #             for a2 in self.alphabet:
    #                 new_word.append(a2)
    #                 if tuple(new_word) not in self.known_mistakes:
    #                     self.update_known(tuple(new_word))
    #                 for a3 in self.alphabet:
    #                     new_word.append(a3)
    #                     if tuple(new_word) not in self.known_mistakes:
    #                         self.update_known(tuple(new_word))
    #                     for a4 in self.alphabet:
    #                         new_word.append(a4)
    #                         if tuple(new_word) not in self.known_mistakes:
    #                             self.update_known(tuple(new_word))
    #                         for a5 in self.alphabet:
    #                             new_word.append(a5)
    #                             if tuple(new_word) not in self.known_mistakes:
    #                                 self.update_known(tuple(new_word))
    #                         new_word.pop(-1)
    #                         new_word.pop(-1)
    #                     new_word.pop(-1)
    #                     new_word.pop(-1)
    #                 new_word.pop(-1)
    #                 new_word.pop(-1)
    #             # new_word.pop(-1)
    #
    #             # new_word.append(a4)
    #             # if tuple(new_word) not in self.known_mistakes:
    #             #     self.update_known(tuple(new_word))
    #
    #         # while count < 200:
    #         # # for _ in range(5000):
    #         #     new_word = scrumble_word_reducing(word, self.alphabet,
    #         #                                       self.scrumble_prob, self.scrumble_prob_inc)
    #         #     if new_word in self.known_mistakes:
    #         #         continue
    #         #     count += 1
    #         #     state = self.init_state
    #         #     for letter in new_word:
    #         #         state = self.transitions[state][letter]
    #         #     label = state in self.final_states
    #         #     self.known_mistakes.update({new_word: not label})
    #         # # print("after len")
    #         # # print(count)
    #         # # print(len(self.known_mistakes))
    #         # print("after:{}".format(len(self.known_mistakes)))
    #         return self.known_mistakes[word]
    #     else:
    #         self.known_mistakes.update({word: label_out})
    #         return label_out
    #
    def update_known(self, new_word):
        state = self.init_state
        for letter in new_word:
            state = self.transitions[state][letter]
        label = state in self.final_states
        self.known_mistakes.update({new_word: not label})
