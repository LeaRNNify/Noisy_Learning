import logging
import time
from collections import namedtuple

# import matplotlib.pyplot as plt
import numpy as np

from counter_dfa import CounterDFA, NoisyCounterDFA, DFAFinalCount
from dfa import DFA, DFANoisy
from dfa_check import DFAChecker
from learner_decison_tree import DecisionTreeLearner
from noisy_input_dfa import NoisyInputDFA
from random_words import random_word, confidence_interval_many_for_reuse, confidence_interval_many_cython, \
    confidence_interval_many_for_reuse_2, confidence_interval_single
from teacher import Teacher
from exact_teacher import ExactTeacher
from randwords import is_words_in_dfa, compare_list_of_bool, is_words_in_counterDfa, \
    is_words_in_dfa_finalcount


def random_words(batch_size, alphabet, word_length):
    return [random_word(alphabet, 1 / word_length) for _ in range(batch_size)]

# def minimize_dfa(dfa: DFA) -> DFA:
#     teacher_pac = ExactTeacher(dfa)
#     student = DecisionTreeLearner(teacher_pac)
#     teacher_pac.teach(student)
#     return student.dfa

class StupidGuess:
    def __init__(self, dic):
        self.dic = dic

    def is_word(self, w):
        if w in self.dic:
            return self.dic[w]
        elif np.random.randint(0, 2) == 0:
            return False
        else:
            return True


class PACTeacher(Teacher):

    def __init__(self, model: DFA, epsilon=0.001, delta=0.001, word_probability=0.01, max_refinements=1):
        assert ((epsilon <= 1) & (delta <= 1))
        Teacher.__init__(self, model)
        self.epsilon = epsilon
        self.delta = delta
        self._log_delta = np.log(delta)
        self._log_one_minus_epsilon = np.log(1 - epsilon)
        self.num_equivalence_asked = 0
        self._word_probability = word_probability
        self.prev_examples = {}
        self.number_of_mq = 0
        self.max_refinements = max_refinements

    def equivalence_query(self, dfa: DFA):
        """
        Tests whether the dfa is equivalent to the model by testing random words.
        If not equivalent returns an example
        """
        number_of_rounds = int(
            (1 / self.epsilon) * (np.log(1 / self.delta) + np.log(2) * (self.num_equivalence_asked + 1)))
        self.num_equivalence_asked = self.num_equivalence_asked + 1
        batch_size = 1000

        for i in range(int(number_of_rounds / batch_size) + 1):
            self.number_of_mq = self.number_of_mq + 1000
            batch = random_words(batch_size, tuple(self.alphabet), int(1 / self._word_probability))

            if isinstance(self.model, CounterDFA):
                mod_words = is_words_in_counterDfa(self.model, batch)
            else:
                mod_words = [self.model.is_word_in(w) for w in batch]

            dfa_words = is_words_in_dfa(dfa, batch)
            for x, y, w in zip(mod_words, dfa_words, batch):
                self.prev_examples[w] = x
                if x != y:
                    return w
                
            
        return None

    def membership_query(self, word):
        self.number_of_mq = self.number_of_mq + 1
        self.prev_examples[word] = self.model.is_word_in(word)
        return self.model.is_word_in(word)

    def teach(self, learner, max_eq=250):
        self.num_equivalence_asked = 0
        learner.teacher = self
        i = 0
        start_time = time.time()
        t100 = start_time
        while True:
            if self.num_equivalence_asked > max_eq:
                return
         
            i = i + 1
            if i % 100 == 0:
                print("this is the {}th round".format(i))
                print("{} time has passed from the begging and {} from the last 100".format(time.time() - start_time,
                                                                                            time.time() - t100))
                t100 = time.time()

            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                break
            #num_of_ref = learner.new_counterexample(counter, self.is_counter_example_in_batches, max_refinements=1)
            num_of_ref = learner.new_counterexample(counter, False, max_refinements=1)
            self.num_equivalence_asked += num_of_ref - 1

    def teach_limited_equivalence_queries(self, learner: DecisionTreeLearner, equivalence_queries):
        equivalence_queries_left = equivalence_queries
        while equivalence_queries_left > 0:
            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                self.num_equivalence_asked += equivalence_queries - equivalence_queries_left
                return True
            num_of_refinements = learner.new_counterexample(counter, False,
                                                            max_refinements=min(self.max_refinements,
                                                                                equivalence_queries_left))
            equivalence_queries_left -= num_of_refinements

        self.num_equivalence_asked += equivalence_queries
        return False

    def teach_acc_noise_dist2(self, learner, max_prev_larger_then_curr=3):
        self.num_equivalence_asked = 0
        confidence_width = 0.01
        prev_dist, min_dist = 1, 1
        min_dfa = None
        samples = None
        answers = None
        while True:
            finished_learning = self.teach_limited_equivalence_queries(learner, 20)

            distances, samples, answers = confidence_interval_many_for_reuse_2([self.model, learner.dfa],
                                                                               random_word,
                                                                               samples=samples,
                                                                               previous_answers=answers,
                                                                               width=confidence_width,
                                                                               confidence=confidence_width)
            new_dist = distances[0][1]
            if new_dist >= prev_dist:
                max_prev_larger_then_curr -= 1
            elif new_dist < min_dist:
                min_dfa = learner.dfa
                min_dist = new_dist
            prev_dist = new_dist

            logging.debug("this is the {}th round with: prev_dist = {}, "
                          "new_dist = {}, min_dist = {}, Retries left: {}".format(self.num_equivalence_asked,
                                                                                  round(prev_dist, 5),
                                                                                  round(new_dist, 5),
                                                                                  round(min_dist, 5),
                                                                                  max_prev_larger_then_curr))

            if finished_learning or max_prev_larger_then_curr < 0:
                learner.dfa = min_dfa
                return
            
    def teach_intermediate_size(self, learner):
        self.num_equivalence_asked = 0
        confidence_width = 0.01
        samples = None

        min_dfa = None
        i = 0
        start_time = time.time()
        t100 = start_time

        while self.num_equivalence_asked<500:

            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                #learner.dfa = min_dfa
                break
            num_of_ref = learner.new_counterexample(counter, False, max_refinements=1)
            self.num_equivalence_asked += num_of_ref
            #min_dfa=minimize_dfa(learner.dfa)
            #print(f"learner state size: {len(min_dfa.states)}")
        return learner.dfa
            
    def teach_fix_round(self, learner):
        self.num_equivalence_asked = 0
        confidence_width = 0.01
        samples = None

        min_dfa = None
        i = 0
        start_time = time.time()
        t100 = start_time

        while self.num_equivalence_asked<500:

            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                #learner.dfa = min_dfa
                break
            num_of_ref = learner.new_counterexample(counter, False, max_refinements=1)
            self.num_equivalence_asked += num_of_ref
            #min_dfa=minimize_dfa(learner.dfa)
            #print(f"learner state size: {len(min_dfa.states)}")
        return learner.dfa
    
    def teach_size_compare(self, learner, noise):
        self.num_equivalence_asked = 0
        confidence_width = 0.01
        prev_dist=1
        prev_size = 0
        #min_size = 1000
        samples = None
        maxround=500
        rfinal=maxround
        i = 0
        #min_dfa = None
        #begin=False
        listDFA=[]

        while self.num_equivalence_asked<maxround:

            if self.num_equivalence_asked / 20 > i:
                i = i + 1
                dfa=learner.dfa
                listDFA.append(dfa)
                
            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                #learner.dfa = min_dfa
                rfinal=self.num_equivalence_asked
                break
            num_of_ref = learner.new_counterexample(counter, False, max_refinements=1)
            self.num_equivalence_asked += num_of_ref
        
        finalDFA=listDFA[-1]
        dis=confidence_interval_single(finalDFA)
        for j in range(len(listDFA)-1):
            models=[listDFA[j], finalDFA]
            output, samples = confidence_interval_many_cython(models, width=0.01, confidence=0.005, word_prob=0.01)
            
            # if samples is None:
            #         output, samples, answers = confidence_interval_many_for_reuse_2([listDFA[j], finalDFA],
            #                                                                         random_word,
            #                                                                         width=confidence_width,
            #                                                                         confidence=confidence_width)
            # else:
            #         output, _, answers = confidence_interval_many_for_reuse_2([listDFA[j], finalDFA],
            #                                                                   random_word, answers, samples=samples,
            #                                                                   width=confidence_width,
            #                                                                   confidence=confidence_width)
            
            #if output[0][1]<noise*0.1:
            if output[0][1]<dis*0.001:
                learner.dfa=listDFA[j]
                return listDFA[-1], dis
        learner.dfa=listDFA[-1]
        return listDFA[-1], dis
             
         
         
 

    def teach_acc_noise_dist(self, learner, max_prev_larger_then_curr=3):
        self.num_equivalence_asked = 0
        confidence_width = 0.01
        prev_dist = 1
        min_dist = 1
        samples = None

        # times that the previous distance was larger than the current one
        count_prev_larger_then_curr = 0

        min_dfa = None
        i = 0
        start_time = time.time()
        t100 = start_time

        while True:
            if count_prev_larger_then_curr >= max_prev_larger_then_curr  or self.num_equivalence_asked>=500:
                logging.debug(max_prev_larger_then_curr)
                logging.debug(time.time() - start_time)
                learner.dfa = min_dfa
                return min_dfa
            # print(i)

            if self.num_equivalence_asked / 20 > i:
                i = i + 1
                if samples is None:
                    output, samples, answers = confidence_interval_many_for_reuse_2([self.model, learner.dfa],
                                                                                    random_word,
                                                                                    width=confidence_width,
                                                                                    confidence=confidence_width)
                else:
                    output, _, answers = confidence_interval_many_for_reuse_2([self.model, learner.dfa],
                                                                              random_word, answers, samples=samples,
                                                                              width=confidence_width,
                                                                              confidence=confidence_width)

                new_dist = output[0][1]
                
                if new_dist >= prev_dist:
                    count_prev_larger_then_curr += 1
                if new_dist < min_dist:
                    min_dfa = learner.dfa
                    min_dist = new_dist
                    logging.debug(len(min_dfa.states))
                # t100 = time.time()
                logging.debug("this is the {}th round with: prev_dist = {}, "
                              "new_dist = {}, min_dist = {}, Num_of_large_delta: {}".format(self.num_equivalence_asked,
                                                                                            round(prev_dist, 5),
                                                                                            round(new_dist, 5),
                                                                                            round(min_dist, 5),
                                                                                            count_prev_larger_then_curr))
                prev_dist = new_dist

            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                learner.dfa = min_dfa
                break
            num_of_ref = learner.new_counterexample(counter, False, max_refinements=1)
            self.num_equivalence_asked += num_of_ref
            #print(f"the equivalence number: {self.num_equivalence_asked}")
            

    def teach_and_trace(self, student, dfa_model, timeout=900):
        num_of_state = 600
        self.num_equivalence_asked = 0
        output, smaples, answers = confidence_interval_many_for_reuse([dfa_model, self.model, student.dfa], random_word,
                                                                      width=0.0006, confidence=0.001)
        dist_to_dfa_vs = []
        dist_to_rnn_vs = []
        num_of_states = []

        dist_to_dfa_vs.append(1)
        dist_to_rnn_vs.append(1)
        num_of_states.append(0)
        # points.append(DataPoint(len(student.dfa.states), output[0, 2], output[1, 2]))

        a = None
        student.teacher = self
        i = 0
        start_time = time.time()
        t100 = start_time
        next_ask = 30
        while True:
            if self.num_equivalence_asked > num_of_state:
                output, _, answers = confidence_interval_many_for_reuse([dfa_model, self.model, student.dfa],
                                                                        random_word, answers, samples=smaples,
                                                                        width=0.0006, confidence=0.001)
                # print("out:")
                # print(output)
                # points.append(DataPoint(len(student.dfa.states), output[0, 2], output[1, 2]))

                dist_to_dfa_vs.append(output[0][1])
                dist_to_rnn_vs.append(output[0][2])
                num_of_states.append(self.num_equivalence_asked)
                break
            i = i + 1
            if i % 100 == 0:
                logging.debug("this is the {}th round".format(i))
                logging.debug(
                    "{} time has passed from the begging and {} from the last 100".format(time.time() - start_time,
                                                                                          time.time() - t100))
                t100 = time.time()
            counter = self.equivalence_query(student.dfa)

            # print("counter = {}".format(counter))
            if counter is None:
                break
            num_of_ref = student.new_counterexample(counter, do_hypothesis_in_batches=False)
            self.num_equivalence_asked += num_of_ref

            if self.num_equivalence_asked > next_ask:
                next_ask += 30
                # print('compute dist')
                output, _, answers = confidence_interval_many_for_reuse([dfa_model, self.model, student.dfa],
                                                                        random_word, answers, samples=smaples,
                                                                        width=0.0006, confidence=0.001)
                # print("out:")
                # print(output)
                # points.append(DataPoint(len(student.dfa.states), output[0, 2], output[1, 2]))

                dist_to_dfa_vs.append(output[0][1])
                dist_to_rnn_vs.append(output[0][2])
                num_of_states.append(self.num_equivalence_asked)
                # print('done compute dist')

        # plt.plot(num_of_states, dist_to_dfa_vs, label="DvD", color='green', linestyle='dashed')
        # plt.title('original dfa vs extracted dfa')
        #
        # plt.plot(num_of_states, dist_to_rnn_vs, label="RvD", )
        # plt.title('rnn vs extracted dfa')
        # plt.legend()
        # plt.figure()
        #
        # #
        # fig = plt.figure(dpi=1200)
        # ax = fig.add_subplot(2, 1, 1)
        #
        # ax.plot(num_of_states, dist_to_dfa_vs, color='blue', lw=2)
        #
        # ax.set_yscale('log')

        # plt.show()
        ra = range(num_of_state)
        # print("-------------------------------")
        # print(dist_to_dfa_vs)
        # print(num_of_states)
        dist_to_dfa_vs = np.interp(ra, num_of_states, dist_to_dfa_vs)
        # print("-------------------------------")
        # print(dist_to_rnn_vs)
        # print(num_of_states)
        dist_to_rnn_vs = np.interp(ra, num_of_states, dist_to_rnn_vs)

        return dist_to_dfa_vs, dist_to_rnn_vs, num_of_states

    def check_and_teach(self, learner, checkers: [DFAChecker], timeout=900):
        learner.teacher = self
        self.num_equivalence_asked = 0
        start_time = time.time()
        Counter_example = namedtuple('Counter_example', ['word', 'is_super'])

        while True:
            if time.time() - start_time > timeout:
                return
            print(time.time() - start_time)

            counter_example = Counter_example(None, None)

            # Searching for counter examples in the spec:
            counters_examples = (Counter_example(checker.check_for_counterexample(learner.dfa), checker.is_super_set)
                                 for checker in checkers)
            for example in counters_examples:
                if example.word is not None:
                    counter_example = example
                    break
            if counter_example.word is not None:
                if counter_example.is_super != (self.model.is_word_in(counter_example.word)):
                    self.num_equivalence_asked += 1
                    num = learner.new_counterexample(counter_example[0], self.is_counter_example_in_batches)
                    if num > 1:
                        self.num_equivalence_asked += num - 1
                else:
                    print('found counter mistake in the model: ', counter_example)
                    return counter_example

            # Searching for counter examples in the the model:
            else:

                counter_example = self.equivalence_query(learner.dfa)
                if counter_example is None:
                    return None
                else:
                    num = learner.new_counterexample(counter_example, self.is_counter_example_in_batches)
                    if num > 1:
                        self.num_equivalence_asked += num - 1

    def teach_a_superset(self, learner, timeout=900):
        self.num_equivalence_asked = 0
        learner.teacher = self
        i = 0
        start_time = time.time()
        t100 = start_time
        while True:
            if time.time() - start_time > timeout:
                print(time.time() - start_time)
                return
            # print(i)
            i = i + 1
            if i % 100 == 0:
                print("this is the {}th round".format(i))
                print("{} time has passed from the begging and {} from the last 100".format(time.time() - start_time,
                                                                                            time.time() - t100))
                t100 = time.time()

            counter = self.model_subset_of_dfa_query(learner.dfa)
            if counter is None:
                break
            learner.new_counterexample(counter, self.is_counter_example_in_batches)
