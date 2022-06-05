from abc import ABC

from dfa import DFA
from model_checker import ModelChecker


class DFAChecker(ModelChecker, ABC):

    def __init__(self, specification: DFA, is_super_set=True):
        super().__init__(specification)
        self.is_super_set = is_super_set

    def check_for_counterexample(self, dfa):
        if self.is_super_set:
            sup_dfa = self.specification
            inf_dfa = dfa
        else:
            sup_dfa = dfa
            inf_dfa = self.specification

        dfs_stack = []
        word_path = ()

        dfs_stack.append([(inf_dfa.init_state, sup_dfa.init_state), word_path])
        visited = []

        while len(dfs_stack) != 0:
            [(model_state, spec_state), word_path] = dfs_stack.pop(0)
            if (model_state, spec_state) in visited:
                continue
            else:
                visited.append((model_state, spec_state))

            if inf_dfa.is_final_state(model_state) and not (sup_dfa.is_final_state(spec_state)):
                return word_path

            for letter in dfa.alphabet:
                dfs_stack.append([(inf_dfa.next_state_by_letter(model_state, letter), \
                                   sup_dfa.next_state_by_letter(spec_state, letter)),
                                  word_path + tuple([letter])])

        return None
