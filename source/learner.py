from teacher import Teacher
from abc import ABC, abstractmethod


class Learner(ABC):

    def __init__(self, teacher: Teacher):
        """
        constructor
        """
        self.teacher = teacher

    @abstractmethod
    def new_counterexample(self, word):
        pass
