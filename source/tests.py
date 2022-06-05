import time
import timeit
import unittest

from dfa import DFA, random_dfa, dfa_intersection
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from pac_teacher import PACTeacher
from counter_dfa import CounterDFA, from_dfa_to_rand_counter_dfa


class Test(unittest.TestCase):

    def test_dfa(self):
        # dfa with the language a*b*
        dfa0 = DFA(1, {1, 2}, {1: {"a": 1, "b": 2},
                               2: {"a": 3, "b": 2},
                               3: {"a": 3, "b": 3}})
        self.assertTrue(dfa0.is_word_in(""))
        self.assertTrue(dfa0.is_word_in("aaaabbb"))
        self.assertFalse(dfa0.is_word_in("aaabaaaabbbbbb"))

        # dfa with the language immediately after every "a" there is a "b"
        dfa1 = DFA(1, {1}, {1: {"a": 2, "b": 1, "c": 1},
                            2: {"a": 3, "b": 1, "c": 3},
                            3: {"a": 3, "b": 3, "c": 3}})
        self.assertTrue(dfa1.is_word_in("abccccbbbbcccabababccccc"))
        self.assertFalse(dfa1.is_word_in("abccccbbabbcccaabababaaaa"))

        dfa2 = DFA(1, {1, 4}, {1: {"a": 2, "b": 1, "c": 1},
                               2: {"a": 3, "b": 1, "c": 3},
                               3: {"a": 3, "b": 3, "c": 4},
                               4: {"a": 3, "b": 3, "c": 4}})

        # dfa with the language immediately after every "a" there is a "b"
        dfa3 = DFA(1, {1, 5}, {1: {"a": 2, "b": 5, "c": 5},
                               2: {"a": 3, "b": 1, "c": 3},
                               3: {"a": 4, "b": 4, "c": 4},
                               4: {"a": 3, "b": 3, "c": 3},
                               5: {"a": 2, "b": 1, "c": 1}})

        self.assertTrue(dfa1 == dfa1)
        self.assertTrue(dfa1 == dfa3)
        self.assertTrue(dfa1 != dfa2)
        self.assertTrue(dfa2.is_language_not_subset_of(dfa1) is not None)

        dfa_0_inter_2 = dfa_intersection(dfa2, dfa1)
        self.assertTrue(dfa_0_inter_2 == dfa3)
        self.assertTrue(dfa_0_inter_2.is_language_not_subset_of(dfa2) is None)
        self.assertTrue(dfa_0_inter_2.is_language_not_subset_of(dfa1) is None)

    def test_learning_algo(self):
        dfa = DFA(1, {1}, {1: {"a": 2, "b": 1, "c": 1},
                           2: {"a": 3, "b": 1, "c": 3},
                           3: {"a": 3, "b": 3, "c": 3}})

        # dfa with the language immediately after every "a" there is a "b"
        dfa2 = DFA(1, {1, 5}, {1: {"a": 2, "b": 5, "c": 5},
                               2: {"a": 3, "b": 1, "c": 3},
                               3: {"a": 4, "b": 4, "c": 4},
                               4: {"a": 3, "b": 3, "c": 3},
                               5: {"a": 2, "b": 1, "c": 1}})

        teacher_exact = ExactTeacher(dfa2)
        student_exact = DecisionTreeLearner(teacher_exact)
        teacher_exact.teach(student_exact)
        self.assertTrue(dfa == student_exact.dfa)

        teacher_pac = PACTeacher(dfa2)
        student_pac = DecisionTreeLearner(teacher_pac)
        while dfa2 != student_pac.dfa:
            teacher_pac.teach(student_pac)

    def test_check_and_teach(self):
        dfa1 = DFA(1, {1}, {1: {"a": 2, "b": 1, "c": 1},
                            2: {"a": 3, "b": 1, "c": 3},
                            3: {"a": 3, "b": 3, "c": 3}})

        dfa2 = DFA(1, {1, 4}, {1: {"a": 2, "b": 1, "c": 1},
                               2: {"a": 3, "b": 1, "c": 3},
                               3: {"a": 3, "b": 3, "c": 4},
                               4: {"a": 3, "b": 3, "c": 4}})

        teacher_pac = PACTeacher(dfa2)
        student_pac = DecisionTreeLearner(teacher_pac)
        self.assertTrue(teacher_pac.check_and_teach(student_pac, [DFAChecker(dfa2)]) is None)

        teacher_pac = PACTeacher(dfa2)
        student_pac = DecisionTreeLearner(teacher_pac)
        self.assertTrue(teacher_pac.check_and_teach(student_pac, [DFAChecker(dfa1)]) is not None)

    def test_rand_long(self):
        # Testing regular extraction:
        for _ in range(2):
            dfa_rand = random_dfa(["a", "b", "c", "d", "e"], max_states=100)
            teacher_exact = ExactTeacher(dfa_rand)
            student_exact = DecisionTreeLearner(teacher_exact)

            init_time = time.time()
            teacher_exact.teach(student_exact)
            print("exact: " + str(time.time() - init_time))
            self.assertTrue(dfa_rand == student_exact.dfa)

            teacher_pac = PACTeacher(dfa_rand)
            student_pac = DecisionTreeLearner(teacher_pac)

            count = -1
            init_time = time.time()
            while dfa_rand != student_pac.dfa:
                count = count + 1
                teacher_pac.teach(student_pac)
            print("PAC: " + str(time.time() - init_time) + ", restarted learning process " + str(count) + " times.")

        # Testing regular extraction:
        for _ in range(2):
            dfa_rand1 = random_dfa(["a", "b", "c", "d", "e"], max_states=50)
            dfa_rand2 = random_dfa(["a", "b", "c", "d", "e"], max_states=50)

            dfa_intersect = dfa_intersection(dfa_rand1, dfa_rand2)

            teacher_pac = PACTeacher(dfa_intersect)
            student_pac = DecisionTreeLearner(teacher_pac)
            self.assertTrue(teacher_pac.check_and_teach(student_pac, [DFAChecker(dfa_rand1)]) is None)

            teacher_pac = PACTeacher(dfa_intersect)
            student_pac = DecisionTreeLearner(teacher_pac)
            self.assertTrue(teacher_pac.check_and_teach(student_pac, [DFAChecker(dfa_rand2)]) is None)

            if not dfa_rand1.is_language_not_subset_of(dfa_rand2):
                teacher_pac = PACTeacher(dfa_rand1)
                student_pac = DecisionTreeLearner(teacher_pac)
                self.assertTrue(teacher_pac.check_and_teach(student_pac, [DFAChecker(dfa_rand2)]) is not None)

    def test_counter_dfa(self):
        dfa = DFA(1, {1, 2}, {1: {"a": 1, "b": 2},
                              2: {"a": 3, "b": 2},
                              3: {"a": 3, "b": 3}})
        sup = False
        alphabet2counter = {"a": -1, "b": 1}
        init_tokens = 2
        cdfa = CounterDFA(dfa.init_state, dfa.final_states, dfa.transitions, alphabet2counter, init_tokens, sup)

        print(cdfa)

        self.assertTrue(cdfa.is_word_in("aa"))
        self.assertFalse(cdfa.is_word_in("aaa"))
        self.assertTrue(cdfa.is_word_in("aaaabbb"))
        self.assertFalse(cdfa.is_word_in("aaaab"))
        self.assertFalse(cdfa.is_word_in("aabbbbba"))
