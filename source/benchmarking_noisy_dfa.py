import datetime
import logging
import os
import time
from itertools import product

import tabulate

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
    def __init__(self,
                 epsilons=(0.005,), deltas=(0.005,), word_probs=(0.01,), max_eq=250,
                 max_extracted_dfa_worsen_distance=5,
                 p_noise=(0.01, 0.005, 0.0025, 0.0015, 0.001), dfa_noise=DFANoisy,
                 min_dfa_state=20, max_dfa_states=60, max_alphabet_size=20, min_alphabet_size=4,
                 dist_epsilon_delta=0.005):
        """

            @param epsilons: epsilons for the PAC equivalence query
            @param deltas: deltas for the PAC equivalence query
            @param word_probs: the probability used in the geometrical random variable used to generate random words
            @param max_eq: the maximal number of equivalence queries used to extract the DFA
            @param max_extracted_dfa_worsen_distance: while learning we check the distance of the currently extracted
                   dfa to the target on, if the distance between measures happens more times than this number we stop
                   the extraction.

            @param p_noise: the amount of noise we add when not using CounterDFA
            @param dfa_noise: the type of noise

            @param min_dfa_state: the maximal number of states in the random DFA
            @param max_dfa_states: the minimal number of states in the random DFA
            @@param max_alphabet_size: the maximal size of the alphabet used in the random DFA
            @@param min_alphabet_size: the minimal size of the alphabet used in the random DFA

            @param dist_epsilon_delta: the epsilon and delta used for computing the distance of the models
            """
        # PAC learning properties
        self.deltas = deltas
        self.word_probs = word_probs
        self.epsilons = epsilons
        self.max_eq = max_eq
        self.max_extracted_dfa_worsen_distance = max_extracted_dfa_worsen_distance

        # Noisy DFA properties
        self.p_noise = p_noise
        self.dfa_noise = dfa_noise

        # Random dfa properties
        self.min_alphabet_size = min_alphabet_size
        self.max_alphabet_size = max_alphabet_size
        self.max_dfa_states = max_dfa_states
        self.min_dfa_states = min_dfa_state

        # Result distance computation
        self.dist_epsilon_delta = dist_epsilon_delta

    def benchmarks_noise_model(self, num_benchmarks=10, save_dir=None):
        """
        Runs the noisy learning benchmarks and produces a summary of these results
        @param num_benchmarks: the number of benchmarks to generate for each p_noise, epsilons, deltas, word_probs
        @param save_dir: the directory to save the results if none then creates a new one
        """
        if save_dir is None:
            save_dir = "../models/random_bench_noisy_dfa_{}".format(
                datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
            os.makedirs(save_dir)

        benchmarks: pd.DataFrame = pd.DataFrame()

        for bench_num in range(1, num_benchmarks + 1):
            start_time = time.time()
            logging.info("Running benchmark {}/{}".format(bench_num, num_benchmarks))
            benchmark_list = self.rand_benchmark(
                save_dir + "/" + format(datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S")) + str(bench_num))
            logging.debug("Summary for the {}th benchmark".format(bench_num))
            logging.debug(benchmark_list)

            benchmarks = pd.concat([benchmarks, pd.DataFrame.from_records(benchmark_list)])
            logging.info(f"Finished benchmark {bench_num}/{num_benchmarks}, in {time.time() - start_time} sec")

        benchmarks.reset_index()
        benchmarks.to_csv(save_dir + "/results.csv")
        self.benchmark_summary(benchmarks, save_dir)

    @staticmethod
    def benchmark_summary(benchmarks: pd.DataFrame, save_dir: str):
        """
        writes a summary of the benchmarks results
        @param benchmarks: The benchmarks results
        @param save_dir: where to save the results
        """
        summary_lines = []

        if benchmarks['noise_type'].iloc[0] == "noisy_DFA":
            benchmarks['dist_dfa_noisy'] = benchmarks['mistake_prob']

        distances = [1, 0.025, 0.005, 0.002, 0.001, 0.0005]

        for range_min, range_max in zip(distances[1:], distances[0:-2]):
            benchmarks_p: pd.DataFrame = benchmarks.loc[(benchmarks['dist_dfa_noisy'] <= range_max) &
                                                        (benchmarks['dist_dfa_noisy'] > range_min)]
            summary_lines.append(
                {'range_min': range_min,
                 'range_max': range_max,
                 'Dist to Noisy': benchmarks_p['dist_dfa_noisy'].mean(),
                 'Dist to Extracted': benchmarks_p['dist_dfa_extr'].mean(),
                 'Dist Noisy to Extracted': benchmarks_p['dist_noisy_extr'].mean(),
                 'Gain': (benchmarks_p['dist_dfa_noisy'].mean() / benchmarks_p['dist_dfa_extr'].mean()),
                 'STD Original to Extracted': benchmarks_p['dist_dfa_extr'].std()
                 })

        summary: pd.DataFrame = pd.DataFrame.from_records(summary_lines)
        summary.to_csv(save_dir + "/results_summary.csv")
        logging.info("Summary of run: \n" + summary.to_markdown())

    def rand_benchmark(self, save_dir=None):
        base_benchmark_summary = {}

        alphabet = self.generate_random_alphabet(base_benchmark_summary)

        dfa = self.generate_random_dfa_and_save(alphabet, save_dir)
        base_benchmark_summary.update(
            {"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states), 'dfa_unique_session_id': dfa.id})

        logging.debug(f"DFA to learn {dfa}")

        benchmarks = self.extract_measure(dfa, base_benchmark_summary, save_dir)

        return benchmarks

    def generate_random_dfa_and_save(self, alphabet, save_dir):
        """
        generates a random DFA according to the min_dfa_states, max_dfa_states and the alphabet, and saves it
        @param alphabet:
        @param save_dir:
        @return: the random DFA
        """
        while True:
            dfa_rand = random_dfa(alphabet, min_state=self.min_dfa_states, max_states=self.max_dfa_states)
            dfa = minimize_dfa(dfa_rand)
            if len(dfa.states) > self.min_dfa_states:
                break

        if save_dir is not None:
            save_dfa_as_part_of_model(save_dir, dfa, name="dfa")
        return dfa

    def extract_measure(self, dfa: DFA, base_benchmark: dict, dir_name=None):
        benchmarks = []
        # todo: Need to maybe add back num_of_retry = 1.
        for p_noise, epsilon, word_prob in product(self.p_noise, self.epsilons, self.word_probs):
            logging.debug(f"Running p_noise = {p_noise}:")
            benchmark = base_benchmark.copy()
            benchmark.update({'epsilon': epsilon, 'max_EQ': self.max_eq, 'word_prob': word_prob})
            suffix = "EpDel-" + str(epsilon) + "MaxEQ" + str(self.max_eq) + "WProb" + str(word_prob)
            if self.dfa_noise == CounterDFA:
                dfa_noisy = close_rand_counter_dfa(dfa)
            else:
                dfa_noisy = self.dfa_noise(dfa.init_state, dfa.final_states, dfa.transitions, mistake_prob=p_noise)
            models = [dfa, dfa_noisy]

            if type(dfa_noisy) == CounterDFA:
                benchmark['noise_type'] = "counter_DFA"
            elif type(dfa_noisy) == DFANoisy:
                benchmark.update({'noise_type': 'noisy_DFA', "mistake_prob": dfa_noisy.mistake_prob})
            else:
                benchmark.update({'noise_type': 'noisy_input_DFA', "mistake_prob": dfa_noisy.mistake_prob})

            extracted_dfa = self.extract_dfa(dfa_noisy, benchmark, word_prob,
                                             epsilon=epsilon, delta=epsilon)
            if dir_name is not None:
                save_dfa_as_part_of_model(dir_name, extracted_dfa,
                                          name="extracted_dfa_p" + suffix,
                                          force_overwrite=True)
            models.append(extracted_dfa)
            self.compute_distances(models, benchmark, epsilon=self.dist_epsilon_delta,
                                   word_prob=word_prob)
            benchmarks.append(benchmark)
        return benchmarks

    def generate_random_alphabet(self, benchmark_summary):
        full_alphabet = "abcdefghijklmnopqrstuvwxyz"
        alphabet = full_alphabet[0:np.random.randint(self.min_alphabet_size, self.max_alphabet_size)]
        benchmark_summary.update({"alph_len": len(alphabet)})
        return alphabet

    def extract_dfa(self, dfa, benchmark, word_probability=0.001, epsilon=0.001, delta=0.001):
        teacher_pac = PACTeacher(dfa, epsilon, delta, word_probability=word_probability)
        logging.debug("Starting DFA extraction")

        start_time = time.time()
        student = DecisionTreeLearner(teacher_pac)

        teacher_pac.teach_acc_noise_dist(student, self.max_extracted_dfa_worsen_distance)
        logging.debug(student.dfa)
        benchmark.update({"extraction_time": time.time() - start_time})
        benchmark.update({"extraction_loops": teacher_pac.num_equivalence_asked})
        benchmark.update({"membership_query": teacher_pac.number_of_mq})
        logging.debug("time = {}".format(time.time() - start_time))
        dfa_extract = minimize_dfa(student.dfa)
        logging.debug(dfa_extract)
        benchmark.update({"dfa_extract_states": len(dfa_extract.states),
                          "dfa_extract_final": len(dfa_extract.final_states)})

        return dfa_extract

    @staticmethod
    def compute_distances(models, benchmark, epsilon=0.01, delta=0.005, word_prob=0.01):
        start_time = time.time()
        logging.debug("Starting distance measuring")
        output, samples = confidence_interval_many_cython(models, width=epsilon, confidence=delta, word_prob=word_prob)
        logging.debug("The confidence interval for epsilon = {} , delta = {}".format(epsilon, delta))
        logging.debug(output)
        dist_2_original = np.average(output[0][2:])
        dist_2_noisy = np.average(output[1][2:])
        benchmark.update({"dist_dfa_noisy": output[0][1],
                          "dist_dfa_extr": dist_2_original,
                          "dist_noisy_extr": dist_2_noisy})

        logging.debug(output)

        logging.debug("Finished distance measuring in {}'s".format(time.time() - start_time))
