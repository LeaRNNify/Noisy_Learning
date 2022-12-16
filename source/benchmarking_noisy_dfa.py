import csv
import datetime
import os
import time

import numpy as np
import pandas as pd

from counter_dfa import from_dfa_to_rand_counter_dfa, CounterDFA
from dfa import DFA, random_dfa, save_dfa_as_part_of_model, DFANoisy
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from pac_teacher import PACTeacher
from random_words import confidence_interval_many_cython


def minimize_dfa(dfa: DFA) -> DFA:
    teacher_pac = ExactTeacher(dfa)
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student)
    return student.dfa


def close_rand_counter_dfa(dfa):
    dfa_counter = None
    for _ in range(20):
        dfa_counter = from_dfa_to_rand_counter_dfa(dfa)
        p, _ = (confidence_interval_many_cython([dfa, dfa_counter], 0.001, 0.005))
        print(p)
        if 0.00005 < p[1][0] < 0.01:
            return dfa_counter

    return dfa_counter


class BenchmarkingNoise:
    def __init__(self, epsilons=(0.005,), p_noise=(0.01, 0.005, 0.0025, 0.0015, 0.001), dfa_noise=DFANoisy,
                 word_probs=(0.01,), max_eq=250):

        self.word_probs = word_probs
        self.dfa_noise = dfa_noise
        self.p_noise = p_noise
        self.epsilons = epsilons
        self.max_eq = 250

    def benchmarks_noise_model(self, num_of_bench=10, save_dir=None):
        benchmarks: pd.DataFrame = pd.DataFrame()
        if save_dir is None:
            save_dir = "../models/random_bench_noisy_dfa_{}".format(
                datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
            os.makedirs(save_dir)

        for num in range(1, num_of_bench + 1):
            print("Running benchmark {}/{}:".format(num, num_of_bench))
            benchmark_list = self.rand_benchmark(
                save_dir + "/" + format(datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S")) + str(num))
            print("Summary for the {}th benchmark".format(num))
            print(benchmark_list)

            benchmarks = pd.concat([benchmarks, pd.DataFrame.from_records(benchmark_list)])

        benchmarks.to_csv(save_dir + "/results.csv")
        self.benchmark_summary(benchmarks,save_dir)

    @staticmethod
    def benchmark_summary(benchmarks: pd.DataFrame, save_dir: str):
        summary_lines = []
        for p in benchmarks['mistake_prob']:
            if benchmarks['noise_type'].iloc[0] == "noisy_dfa":
                benchmarks['dist_dfa_noisy'] = p

            benchmarks_p = benchmarks.loc[benchmarks['mistake_prob'] == p]
            summary_lines.append({'mistake_prob': benchmarks_p['mistake_prob'],
                                  'dist_dfa_noisy': benchmarks_p['dist_dfa_noisy'].mean(),
                                  'dist_dfa_extr': benchmarks_p['dist_dfa_extr'].mean(),
                                  'dist_noisy_extr': benchmarks_p['dist_noisy_extr'].mean(),
                                  'gain': (benchmarks_p['dist_dfa_noisy'].mean() / benchmarks_p[
                                      'dist_dfa_extr']).mean()})
        pd.DataFrame.from_records(summary_lines).to_csv(save_dir + "/results_summary.csv")

        pass

    def rand_benchmark(self, save_dir=None):
        full_alphabet = "abcdefghijklmnopqrstuvwxyz"

        alphabet = full_alphabet[0:np.random.randint(4, 20)]
        # alphabet = full_alphabet[0:19]

        base_benchmark = {}
        base_benchmark.update({"alph_len": len(alphabet)})
        # todo minstate = 20 max_states = 60
        while True:
            dfa_rand = random_dfa(alphabet, min_state=5, max_states=10)
            dfa = minimize_dfa(dfa_rand)
            # todo > 20:
            if len(dfa.states) > 5:
                break

        base_benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

        if save_dir is not None:
            save_dfa_as_part_of_model(save_dir, dfa, name="dfa")

        print("DFA to learn {}".format(dfa))

        benchmarks = self.extract_measure(dfa, base_benchmark, save_dir)

        return benchmarks

    def extract_measure(self, dfa: DFA, base_benchmark: dict, dir_name=None):
        benchmarks = []
        num_of_retry = 1
        for p in self.p_noise:
            print("Running p = {}:".format(p))
            if self.dfa_noise == CounterDFA:
                dfa_noisy = close_rand_counter_dfa(dfa)
            else:
                dfa_noisy = self.dfa_noise(dfa.init_state, dfa.final_states, dfa.transitions, mistake_prob=p)
            for epsilon in self.epsilons:
                for word_prob in self.word_probs:
                    benchmark = base_benchmark.copy()
                    models = [dfa, dfa_noisy]
                    for _ in range(num_of_retry):
                        benchmark.update({'epsilon': epsilon, 'max_EQ': self.max_eq, 'word_prob': word_prob})
                        suffix = "EpDel-" + str(epsilon) + "MaxEQ" + str(self.max_eq) + "WProb" + str(word_prob)
                        if type(dfa_noisy) == CounterDFA:
                            benchmark['noise_type'] = "counter_DFA"
                        elif type(dfa_noisy) == DFANoisy:
                            benchmark.update({'noise_type': 'noisy_DFA', "mistake_prob": dfa_noisy.mistake_prob})
                        else:
                            benchmark.update({'noise_type': 'noisy_DFA', "mistake_prob": dfa_noisy.mistake_prob})

                        extracted_dfa = self.extract_dfa(dfa_noisy, benchmark, word_prob,
                                                         epsilon=epsilon, delta=epsilon)

                        if dir_name is not None:
                            save_dfa_as_part_of_model(dir_name, extracted_dfa,
                                                      name="extracted_dfa_p" + suffix + "-3",
                                                      force_overwrite=True)
                        models.append(extracted_dfa)
                    self.compute_distances(models, benchmark, epsilon=0.05,
                                           word_prob=word_prob)
                    benchmarks.append(benchmark)
        return benchmarks

    @staticmethod
    def extract_dfa(dfa, benchmark, word_probability=0.001, epsilon=0.001, delta=0.001):
        teacher_pac = PACTeacher(dfa, epsilon, delta, word_probability=word_probability)
        print("Starting DFA extraction")
        ###################################################
        # Doing the model checking after a DFA extraction
        ###################################################
        start_time = time.time()
        student = DecisionTreeLearner(teacher_pac)

        # todo : this needs to be 5:
        teacher_pac.teach_acc_noise_dist(student, 1)
        print(student.dfa)
        benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})
        benchmark.update({"extraction_loops": teacher_pac.num_equivalence_asked})
        benchmark.update({"membership_query": teacher_pac.number_of_mq})
        print("time = {}".format(time.time() - start_time))
        dfa_extract = minimize_dfa(student.dfa)
        print(dfa_extract)
        benchmark.update({"dfa_extract_states": len(dfa_extract.states),
                          "dfa_extract_final": len(dfa_extract.final_states)})

        return dfa_extract

    @staticmethod
    def compute_distances(models, benchmark, epsilon=0.01, delta=0.005, word_prob=0.01):
        start_time = time.time()
        print("Starting distance measuring")
        output, samples = confidence_interval_many_cython(models, width=epsilon, confidence=delta, word_prob=word_prob)
        print("The confidence interval for epsilon = {} , delta = {}".format(epsilon, delta))
        print(output)
        dist_2_original = np.average(output[0][2:])
        dist_2_noisy = np.average(output[1][2:])
        benchmark.update({"dist_dfa_noisy": output[0][1],
                          "dist_dfa_extr": dist_2_original,
                          "dist_noisy_extr": dist_2_noisy})

        print(output)

        print("Finished distance measuring in {}'s".format(time.time() - start_time))
