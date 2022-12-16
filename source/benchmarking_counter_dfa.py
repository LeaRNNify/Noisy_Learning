import csv
import datetime
import os
import time

import numpy as np

from counter_dfa import from_dfa_to_rand_counter_dfa, from_counter_to_noisy_counter, from_dfa_to_dfa_final_count
from dfa import DFA, random_dfa, dfa_intersection, save_dfa_as_part_of_model, DFANoisy
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from pac_teacher import PACTeacher
from random_words import confidence_interval_many, random_word, confidence_interval_subset, confidence_interval, \
    confidence_interval_many_cython

FIELD_NAMES = ["alph_len",

               "dfa_states", "dfa_final",
               "dfa_extract_states", "dfa_extract_final",

               "extraction_time",

               "dist_dfa_vs_noisy", "dist_dfa_vs_extr", "dist_noisy_vs_extr"]


def write_csv_header(filename, fieldnames=None):
    if fieldnames is None:
        fieldnames = FIELD_NAMES
    with open(filename, mode='a') as employee_file:
        writer = csv.DictWriter(employee_file, fieldnames=fieldnames)
        writer.writeheader()


def write_line_csv(filename, benchmark, fieldnames=None):
    if fieldnames is None:
        fieldnames = FIELD_NAMES
    with open(filename, mode='a') as benchmark_summary:
        writer = csv.DictWriter(benchmark_summary, fieldnames=fieldnames)
        writer.writerow(benchmark)


def minimize_dfa(dfa: DFA) -> DFA:
    teacher_pac = ExactTeacher(dfa)
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student)
    return student.dfa


def extract_mesaure(dfa: DFA, benchmark, dir_name=None):
    # print("here")
    timeout = 300
    dfa_counter = from_dfa_to_rand_counter_dfa(dfa)
    # p, _ = (confidence_interval_many_cython([dfa, dfa_counter], 0.001, 0.005))
    # print(p)
    # if (p[1][0] > 0.1) or (p[1][0] < 0.001):
    #     return
    # print("passed with {}".format(p[1][0]))
    # benchmark.update({"done": 1})

    for epsilon in [0.001]:
        for word_prob in [0.01]:
            # for word_prob in [0.01]:
            suffix = "EpDel-" + str(epsilon) + "TO" + str(timeout) + "WProb" + str(word_prob)
            extracted_dfa = extract_dfa(dfa_counter, benchmark, epsilon, epsilon,
                                        suffix=suffix, timeout=timeout, word_probability=word_prob)
            if dir_name is not None:
                save_dfa_as_part_of_model(dir_name, extracted_dfa, name="extracted_dfa" + suffix + "counter")

            models = [dfa, dfa_counter, extracted_dfa]

            compute_distances(models, benchmark, suffix=suffix, epsilon=0.0005, word_prob=word_prob)

            # noisy counter
            noisy_counter = from_counter_to_noisy_counter(dfa_counter, 0.5)

            suffix = "EpDel-" + str(epsilon) + "TO" + str(timeout) + "WProb" + str(word_prob) + "noise-0.5"
            print("len before: {}".format(len(noisy_counter.known_words)))
            extracted_dfa2 = extract_dfa(noisy_counter, benchmark, epsilon, epsilon,
                                         suffix=suffix, timeout=timeout, word_probability=word_prob)
            print("len after: {}".format(len(noisy_counter.known_words)))
            if dir_name is not None:
                save_dfa_as_part_of_model(dir_name, extracted_dfa2, name="extracted_dfa" + suffix)

            models = [dfa, noisy_counter, extracted_dfa2]

            noisy_counter = from_counter_to_noisy_counter(dfa_counter, 0.2)
            compute_distances(models, benchmark, suffix=suffix, epsilon=0.0005, word_prob=word_prob)

            suffix = "EpDel-" + str(epsilon) + "TO" + str(timeout) + "WProb" + str(word_prob) + "noise-0.2"
            print("len before: {}".format(len(noisy_counter.known_words)))
            extracted_dfa2 = extract_dfa(noisy_counter, benchmark, epsilon, epsilon,
                                         suffix=suffix, timeout=timeout, word_probability=word_prob)
            print("len after: {}".format(len(noisy_counter.known_words)))
            if dir_name is not None:
                save_dfa_as_part_of_model(dir_name, extracted_dfa2, name="extracted_dfa" + suffix)

            models = [dfa, noisy_counter, extracted_dfa2]

            compute_distances(models, benchmark, suffix=suffix, epsilon=0.0005, word_prob=word_prob)
            # extracted_dfa = extract_dfa(dfa_counter, benchmark, timeout=600)
            # if dir_name is not None:
            #     save_dfa_as_part_of_model(dir_name, extracted_dfa, name="extracted_dfa")
            #
            # models = [dfa, dfa_counter, extracted_dfa]
            #
            # compute_distances(models, benchmark)

    benchmark.update({"init_tokens": str(dfa_counter.init_tokens),
                      "CDFA - sup ": dfa_counter.sup,
                      "alph2int": dfa_counter.alphabet2counter})


def extract_mesaure_fc(dfa: DFA, benchmark, dir_name=None):
    # print("here")
    timeout = 300

    for tresh in {1 + 10 ** (-3), 1 + 10 ** (-4)}:
        for mult in {0.5}:
            for second_add in [0.01, 0.001]:
                dfa_c_f = from_dfa_to_dfa_final_count(dfa, mult, tresh, second_add)
                # p, _ = (confidence_interval_many_cython([dfa, dfa_c_f], 0.001, 0.005))
                # print(p)
                # if (p[1][0] > 0.1) or (p[1][0] < 0.0001):
                #     return
                # print("passed with {}".format(p[1][0]))
                benchmark.update({"done": 1})

                for epsilon in [0.001]:
                    for word_prob in [0.01]:
                        # for word_prob in [0.01]:
                        suffix = "tresh-{} mult-{} second_add -{}".format(tresh, mult, second_add)
                        # suffix = "EpDel-" + str(epsilon) + "TO" + str(timeout) + "WProb" + str(word_prob)
                        extracted_dfa = extract_dfa(dfa_c_f, benchmark, epsilon, epsilon,
                                                    suffix=suffix, timeout=timeout, word_probability=word_prob)
                        if dir_name is not None:
                            save_dfa_as_part_of_model(dir_name, extracted_dfa, name="extracted_dfa" + suffix + "fc")

                        models = [dfa, dfa_c_f, extracted_dfa]

                        compute_distances(models, benchmark, suffix=suffix, epsilon=0.005, word_prob=word_prob)

                benchmark.update({"EpDel": epsilon,
                                  "TO": timeout,
                                  "WProb": word_prob})


def extract_dfa(dfa, benchmark, epsilon=0.001, delta=0.001, suffix="", timeout=900, word_probability=0.001):
    teacher_pac = PACTeacher(dfa, epsilon, delta, word_probability=word_probability)

    print("Starting DFA extraction")
    ###################################################
    # Doing the model checking after a DFA extraction
    ###################################################
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach_acc_noise_dist(student)
    benchmark.update({"extraction_time" + suffix: "{:.3}".format(time.time() - start_time)})
    benchmark.update({"extraction_loops" + suffix: teacher_pac.num_equivalence_asked})
    print("time = {}".format(time.time() - start_time))
    dfa_extract = minimize_dfa(student.dfa)
    print(student.dfa)
    benchmark.update({"dfa_extract_states" + suffix: len(dfa_extract.states),
                      "dfa_extract_final" + suffix: len(dfa_extract.final_states)})

    return dfa_extract


def compute_distances(models, benchmark, epsilon=0.001, delta=0.005, suffix="", word_prob=0.01):
    startime = time.time()
    print("Starting distance measuring")
    output, samples = confidence_interval_many_cython(models, width=epsilon, confidence=delta, word_prob=word_prob)
    print("The confidence interval for epsilon = {} , delta = {}".format(epsilon, delta))
    print(output)

    benchmark.update({"dist_dfa_vs_counter" + suffix: "{}".format(output[0][1])
                      # })
                        ,"dist_dfa_vs_extr" + suffix: "{}".format(output[0][2]),
                        "dist_counter_vs_extr" + suffix: "{}".format(output[1][2])})

    print("Finished distance measuring in {}'s".format(time.time() - startime))


def rand_benchmark(save_dir=None):
    full_alphabet = "abcdefghijklmnopqrstuvwxyz"

    alphabet = full_alphabet[0:np.random.randint(4, 20)]
    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    max_final = np.random.randint(6, 40)

    dfa_rand = random_dfa(alphabet, max_states=50)
    dfa = minimize_dfa(dfa_rand)

    benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

    if save_dir is not None:
        save_dfa_as_part_of_model(save_dir, dfa, name="dfa", force_overwrite=True)

    print("DFA to learn {}".format(dfa))

    extract_mesaure(dfa, benchmark, save_dir)

    return benchmark


def run_rand_benchmarks_counter_dfa(num_of_bench=10, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_fc_dfa_{}".format(
            datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)
    first = True
    num = 1
    while num_of_bench >= num:

        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num))
        print(benchmark.keys())
        if "done" not in benchmark.keys():
            continue
        if first:
            first = False
            write_csv_header(save_dir + "/test.csv", benchmark.keys())
            print(benchmark.keys())
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys()
                       )
        num += 1


def dfafinalcount_prob_find():
    first = True
    save_dir = "../models/dfa_final"
    for i in range(20):
        print("Benchmark {}/20 :".format(i))
        full_alphabet = "abcdefghijklmnopqrstuvwxyz"
        alphabet = full_alphabet[0:np.random.randint(4, 20)]
        benchmark = {}
        benchmark.update({"alph_len": len(alphabet)})
        max_final = np.random.randint(6, 40)
        dfa_rand = random_dfa(alphabet, max_states=50)
        dfa = minimize_dfa(dfa_rand)

        for tresh in {1 + 10 ** (-1), 1 + 10 ** (-2), 1 + 10 ** (-3), 1 + 10 ** (-4)}:
            for mult in {0.5, 0.45, 0.4}:
                for second_add in [0.1, 0.01, 0.001]:
        # for tresh in {1 + 10 ** (-3)}:
        #     for mult in {0.5}:
        #         for second_add in [0.1]:
                    print("---------------------")
                    print("doing tresh: {} and  mult: {} and second add = {}".format(tresh, mult, second_add))
                    print("---------------------")
                    benchmark.update({"states": len(dfa.states), "final_states": len(dfa.final_states)})
                    dfa_c_f = from_dfa_to_dfa_final_count(dfa, mult, tresh, second_add)
                    print("init counter:")
                    print(dfa_c_f.init_count)
                    compute_distances([dfa, dfa_c_f], benchmark,
                                      suffix="tresh-{} mult-{} second_add -{}".format(tresh, mult, second_add),
                                      epsilon=0.001,
                                      word_prob=0.01)
        if first:
            first = False
            write_csv_header(save_dir + "/tresh.csv", benchmark.keys())
            print(benchmark.keys())
        print(benchmark)
        write_line_csv(save_dir + "/tresh.csv", benchmark, benchmark.keys()
                       )


def run_rand_benchmarks_final_count_dfa(num_of_bench=10, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_final_count_dfa{}".format(
            datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)
    first = True
    num = 1
    while num_of_bench >= num:

        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num))
        print(benchmark.keys())
        if "done" not in benchmark.keys():
            continue
        if first:
            first = False
            write_csv_header(save_dir + "/test.csv", benchmark.keys())
            print(benchmark.keys())
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys()
                       )
        num += 1