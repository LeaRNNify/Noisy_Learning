from abc import ABC, abstractmethod

from dfa import DFA


class Teacher(ABC):
    def __init__(self, model: DFA):
        """
        Constructor
        """
        self.model = model
        self.alphabet = model.alphabet

    @abstractmethod
    def membership_query(self, word):
        """
        :param word: a word
        :return: True if the word is in the language, False otherwise
        """
        pass

    @abstractmethod
    def equivalence_query(self, dfa):
        """
        :param dfa: a DFA
        :return: a counterexample if the DFA is not equivalent to the model, None otherwise
        """
        pass


    # @abstractmethod
    # def teach(self, dfa):
    #     raise NotImplementedError()