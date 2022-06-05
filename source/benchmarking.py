import copy
import csv
import datetime
import os
import time

import numpy as np

from dfa import DFA, random_dfa, dfa_intersection, save_dfa_as_part_of_model, load_dfa_dot
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from lstar.Extraction import extract as extract_iclm
from lstar.Tomita_Grammars import tomita_1, tomita_2, tomita_3, tomita_4, tomita_5, tomita_6, tomita_7
from modelPadding import RNNLanguageClasifier
from specifications_for_models import Lang, tomita_1_check_languages, tomita_2_check_languages, \
    tomita_3_check_languages, tomita_5_check_languages, tomita_4_check_languages, tomita_6_check_languages, \
    tomita_7_check_languages
from pac_teacher import PACTeacher
from random_words import confidence_interval_many, random_word, confidence_interval_subset, model_check_random
from functools import partial

FIELD_NAMES = ["alph_len",

               "dfa_inter_states", "dfa_inter_final",
               'dfa_spec_states', 'dfa_spec_final',
               'dfa_extract_specs_states', "dfa_extract_specs_final",
               "dfa_extract_states", "dfa_extract_final",
               "dfa_icml18_states", "dfa_icml18_final",

               "rnn_layers", "rnn_hidden_dim", "rnn_dataset_learning", "rnn_dataset_testing",
               "rnn_testing_acc", "rnn_val_acc", "rnn_time",

               "extraction_time_spec", "extraction_mistake_during",
               "extraction_time", "mistake_time_after", "extraction_mistake_after",
               "extraction_time_icml18",

               "dist_rnn_vs_inter", "dist_rnn_vs_extr", "dist_rnn_vs_extr_spec", "dist_rnn_vs_icml18",
               "dist_inter_vs_extr", "dist_inter_vs_extr_spec", "dist_inter_vs_icml18",

               "dist_specs_rnn", "dist_specs_extract", "dist_specs_extract_w_spec", "statistic_checking_time"]


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


#
def learn_dfa(dfa: DFA, benchmark, hidden_dim=-1, num_layers=-1, embedding_dim=-1, batch_size=-1,
              epoch=-1, num_of_exm_per_length=-1, word_training_length=-1):
    if hidden_dim == -1:
        hidden_dim = len(dfa.states) * 6
    if num_layers == -1:
        num_layers = 3
    if embedding_dim == -1:
        embedding_dim = len(dfa.alphabet) * 2
    if num_of_exm_per_length == -1:
        num_of_exm_per_length = 15000
    if epoch == -1:
        epoch = 10
    if batch_size == -1:
        batch_size = 20
    if word_training_length == -1:
        word_training_length = len(dfa.states) + 5

    start_time = time.time()
    model = RNNLanguageClasifier()
    model.train_a_lstm(dfa.alphabet, dfa.is_word_in,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       embedding_dim=embedding_dim,
                       batch_size=batch_size,
                       epoch=epoch,
                       num_of_exm_per_lenght=num_of_exm_per_length,
                       word_traning_length=word_training_length
                       )

    benchmark.update({"rnn_time": "{:.3}".format(time.time() - start_time),
                      "rnn_hidden_dim": hidden_dim,
                      "rnn_layers": num_layers,
                      "rnn_testing_acc": "{:.3}".format(model.test_acc),
                      "rnn_val_acc": "{:.3}".format(model.val_acc),
                      "rnn_dataset_learning": model.num_of_train,
                      "rnn_dataset_testing": model.num_of_test})

    print("time: {}".format(time.time() - start_time))
    return model


def learn_target(target, alphabet, benchmark, hidden_dim=-1, num_layers=-1, embedding_dim=-1, batch_size=-1,
                 epoch=-1, num_of_examples=-1):
    if hidden_dim == -1:
        hidden_dim = 100
    if num_layers == -1:
        num_layers = 3
    if embedding_dim == -1:
        embedding_dim = len(alphabet) * 2
    if epoch == -1:
        epoch = 10
    if batch_size == -1:
        batch_size = 20
    if num_of_examples == -1:
        num_of_examples = 50000

    start_time = time.time()
    model = RNNLanguageClasifier()
    model.train_a_lstm(alphabet, target, random_word,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       embedding_dim=embedding_dim,
                       batch_size=batch_size,
                       epoch=epoch,
                       num_of_examples=num_of_examples
                       )

    benchmark.update({"rnn_time": "{:.3}".format(time.time() - start_time),
                      "rnn_hidden_dim": hidden_dim,
                      "rnn_layers": num_layers,
                      "rnn_testing_acc": "{:.3}".format(model.test_acc),
                      "rnn_val_acc": "{:.3}".format(model.val_acc),
                      "rnn_dataset_learning": model.num_of_train,
                      "rnn_dataset_testing": model.num_of_test})

    print("time: {}".format(time.time() - start_time))
    return model


def learn_and_check(dfa: DFA, spec: [DFAChecker], benchmark, dir_name=None):
    rnn = learn_dfa(dfa, benchmark, epoch=3, num_of_exm_per_length=2000)

    extracted_dfas = check_rnn_acc_to_spec(rnn, spec, benchmark)
    if dir_name is not None:
        rnn.save_lstm(dir_name)
        for extracted_dfa, name in extracted_dfas:
            if isinstance(name, DFA):
                save_dfa_as_part_of_model(dir_name, extracted_dfa, name=name)
            # dfa_extract.draw_nicely(name="_dfa_figure", save_dir=dir_name)

    models = [dfa, rnn, extracted_dfas[0][0], extracted_dfas[1][0], extracted_dfas[2][0]]

    compute_distances(models, spec[0].specification, benchmark, delta=0.05, epsilon=0.05)


def check_rnn_acc_to_spec(rnn, spec, benchmark, timeout=900):
    teacher_pac = PACTeacher(rnn, epsilon=0.005, delta=0.005)
    student = DecisionTreeLearner(teacher_pac)

    print("Starting DFA extraction")
    ##################################################
    # Doing the model checking during a DFA extraction
    ###################################################
    print("Starting DFA extraction with model checking")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    counter = teacher_pac.check_and_teach(student, spec, timeout=timeout)
    benchmark.update({"during_time_spec": "{:.3}".format(time.time() - start_time)})
    dfa_extract_w_spec = student.dfa
    dfa_extract_w_spec = minimize_dfa(dfa_extract_w_spec)

    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_during": "NAN",
                          "dfa_extract_specs_states": len(dfa_extract_w_spec.states),
                          "dfa_extract_specs_final": len(dfa_extract_w_spec.final_states),
                          "dfa_extract_spec_mem_queries": rnn.num_of_membership_queries})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_mistake_during": counter[0],
                          "dfa_extract_specs_states": len(dfa_extract_w_spec.states),
                          "dfa_extract_specs_final": len(dfa_extract_w_spec.final_states),
                          "dfa_extract_spec_mem_queries": rnn.num_of_membership_queries})

    ###################################################
    # Doing the model checking after a DFA extraction
    ###################################################
    print("Starting DFA extraction w/o model checking")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student, max_eq=timeout)
    # benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})

    print("Model checking the extracted DFA")
    counter = student.dfa.is_language_not_subset_of(spec[0].specification)
    if counter is not None:
        if not rnn.is_word_in(counter):
            counter = None

    benchmark.update({"mistake_time_extraction": "{:.3}".format(time.time() - start_time)})

    dfa_extract = minimize_dfa(student.dfa)
    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_after": "NAN",
                          "dfa_extract_states": len(dfa_extract.states),
                          "dfa_extract_final": len(dfa_extract.final_states),
                          "dfa_extract_mem_queries": rnn.num_of_membership_queries})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_mistake_after": counter,
                          "dfa_extract_states": len(dfa_extract.states),
                          "dfa_extract_final": len(dfa_extract.final_states),
                          "dfa_extract_mem_queries": rnn.num_of_membership_queries})

    ###################################################
    # Doing the model checking acc. of a sup lang extraction
    ###################################################
    print("Starting DFA extraction super w/o model checking")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach_a_superset(student, timeout=timeout)
    # benchmark.update({"extraction_super_time": "{:.3}".format(time.time() - start_time)})

    print("Model checking the extracted DFA")
    counter = student.dfa.is_language_not_subset_of(spec[0].specification)
    if counter is not None:
        if not rnn.is_word_in(counter):
            counter = None

    benchmark.update({"mistake_time_super": "{:.3}".format(time.time() - start_time)})

    dfa_extract_super = minimize_dfa(student.dfa)
    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_super_mistake_after": "NAN",
                          "dfa_extract_super_states": len(dfa_extract.states),
                          "dfa_extract_super_final": len(dfa_extract.final_states),
                          "dfa_extract_super_mem_queries": rnn.num_of_membership_queries})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_super_mistake_after": counter,
                          "dfa_extract_super_states": len(dfa_extract.states),
                          "dfa_extract_super_final": len(dfa_extract.final_states),
                          "dfa_extract_super_mem_queries": rnn.num_of_membership_queries})

    print("Finished DFA extraction")

    ###################################################
    # Doing the model checking randomly
    ###################################################
    print("starting rand model checking")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    counter = model_check_random(rnn, spec[0].specification, width=0.005, confidence=0.005)
    if counter is None:
        counter = "NAN"
    benchmark.update({"mistake_time_rand": "{:.3}".format(time.time() - start_time),
                      "mistake_rand": counter,
                      "rand_num_queries": rnn.num_of_membership_queries})

    print(benchmark)
    return (dfa_extract_w_spec, "dfa_extract_W_spec"), \
           (dfa_extract, "dfa_extract"), \
           (dfa_extract_super, "dfa_extract_super")


def check_rnn_acc_to_spec_only_mc(rnn, spec, benchmark, timeout=900):
    teacher_pac = PACTeacher(rnn, epsilon=0.0005, delta=0.0005)
    student = DecisionTreeLearner(teacher_pac)

    print("Starting DFA extraction")
    ##################################################
    # Doing the model checking during a DFA extraction
    ###################################################
    print("Starting DFA extraction with model checking")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    counter = teacher_pac.check_and_teach(student, spec, timeout=timeout)
    benchmark.update({"during_time_spec": "{:.3}".format(time.time() - start_time)})
    dfa_extract_w_spec = student.dfa
    dfa_extract_w_spec = minimize_dfa(dfa_extract_w_spec)

    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_during": "NAN",
                          "dfa_extract_specs_states": len(dfa_extract_w_spec.states),
                          "dfa_extract_specs_final": len(dfa_extract_w_spec.final_states),
                          "dfa_extract_spec_mem_queries": rnn.num_of_membership_queries})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_mistake_during": counter[0],
                          "dfa_extract_specs_states": len(dfa_extract_w_spec.states),
                          "dfa_extract_specs_final": len(dfa_extract_w_spec.final_states),
                          "dfa_extract_spec_mem_queries": rnn.num_of_membership_queries})

    print(benchmark)
    return (dfa_extract_w_spec, "dfa_extract_W_spec")


def extract_dfa_from_rnn(rnn, benchmark, timeout=900):
    rnn.num_of_membership_queries = 0
    teacher_pac = PACTeacher(rnn)

    ###################################################
    # DFA extraction
    ###################################################
    print("Starting DFA extraction w/o model checking")
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student, max_eq=timeout)
    benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})

    dfa_extract = minimize_dfa(student.dfa)
    print(student.dfa)
    benchmark.update({"dfa_extract_states": len(dfa_extract.states),
                      "dfa_extract_final": len(dfa_extract.final_states),
                      "num_of_mem_quarries_extracted": rnn.num_of_membership_queries})

    # ###################################################
    # # Doing DFA extraction acc. to icml18
    # ###################################################
    # print("Starting DFA extraction acc to iclm18")
    # start_time = time.time()
    #
    # dfa_iclm18 = extract_iclm(rnn, time_limit=timeout, initial_split_depth=10)
    #
    # benchmark.update({"extraction_time_icml18": time.time() - start_time,
    #                   "dfa_icml18_states": len(dfa_iclm18.Q),
    #                   "dfa_icml18_final": len(dfa_iclm18.F)})
    #
    # print("Finished DFA extraction")

    return dfa_extract


def compute_distances(models, dfa_spec, benchmark, epsilon=0.005, delta=0.001):
    print("Starting distance measuring")
    output, samples = confidence_interval_many(models, random_word, width=epsilon, confidence=delta)
    print("The confidence interval for epsilon = {} , delta = {}".format(delta, epsilon))
    print(output)

    benchmark.update({"dist_rnn_vs_inter": "{}".format(output[1][0]),
                      "dist_rnn_vs_extr_spec": "{}".format(output[1][2]),
                      "dist_rnn_vs_extr": "{}".format(output[1][3]),
                      "dist_rnn_vs_icml18": "{}".format(output[1][4])})

    benchmark.update({"dist_inter_vs_extr_spec": "{}".format(output[0][2]),
                      "dist_inter_vs_extr": "{}".format(output[0][3]),
                      "dist_inter_vs_icml18": "{}".format(output[0][4])})

    start_time = time.time()
    a, samples = confidence_interval_subset(models[1], dfa_spec, confidence=epsilon, width=delta)
    benchmark.update({"statistic_checking_time": time.time() - start_time})
    b, _ = confidence_interval_subset(models[2], dfa_spec, samples, epsilon, delta)
    c, _ = confidence_interval_subset(models[3], dfa_spec, samples, epsilon, delta)
    benchmark.update(
        {"dist_specs_rnn": "{}".format(a),
         "dist_specs_extract_w_spec": "{}".format(b),
         "dist_specs_extract": "{}".format(c)})

    print("Finished distance measuring")


def compute_distances_no_model_checking(models, benchmark, epsilon=0.005, delta=0.001):
    print("Starting distance measuring")
    output, samples = confidence_interval_many(models, random_word, width=epsilon, confidence=delta)
    print("The confidence interval for epsilon = {} , delta = {}".format(delta, epsilon))
    print(output)

    benchmark.update({"dist_rnn_vs_target": "{}".format(output[1][0]),
                      "dist_rnn_vs_extr": "{}".format(output[1][2]),
                      "dist_target_vs_extr": "{}".format(output[0][2])})

    print("Finished distance measuring")


def rand_benchmark(save_dir=None):
    dfa_spec, dfa_inter = DFA(0, {0}, {0: {0: 0}}), DFA(0, {0}, {0: {0: 0}})

    full_alphabet = "abcdefghijklmnopqrstuvwxyz"

    alphabet = full_alphabet[0:np.random.randint(4, 5)]
    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    while len(dfa_inter.states) < 5 or len(dfa_spec.states) < 2 or (len(dfa_inter.states) > 25):
        dfa_rand1 = random_dfa(alphabet, max_states=15)
        dfa_rand2 = random_dfa(alphabet, max_states=7)

        dfa_inter = minimize_dfa(dfa_intersection(dfa_rand1, dfa_rand2))
        dfa_spec = minimize_dfa(dfa_rand2)

    benchmark.update({"dfa_inter_states": len(dfa_inter.states), "dfa_inter_final": len(dfa_inter.final_states),
                      "dfa_spec_states": len(dfa_spec.states), "dfa_spec_final": len(dfa_spec.final_states)})

    if save_dir is not None:
        save_dfa_as_part_of_model(save_dir, dfa_inter, name="dfa_intersection")
        dfa_inter.draw_nicely(name="intersection_dfa_figure", save_dir=save_dir)

        save_dfa_as_part_of_model(save_dir, dfa_spec, name="dfa_spec")
        dfa_spec.draw_nicely(name="spec_dfa_figure", save_dir=save_dir)

    print("DFA to learn {}".format(dfa_inter))
    print("Spec to learn {}".format(dfa_spec))

    learn_and_check(dfa_inter, [DFAChecker(dfa_spec)], benchmark, save_dir)

    return benchmark


def run_rand_benchmarks(num_of_bench=10, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_{}".format(datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    write_csv_header(save_dir + "/test.csv")
    for num in range(1, num_of_bench + 1):
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num))
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark)


def learn_multiple_times(dfa, dir_save=None):
    for hidden_dim, num_layers in ((20, 2), (50, 5), (100, 10), (200, 20), (500, 50)):
        benchmarks = {}
        lstm = learn_dfa(dfa, benchmarks,
                         hidden_dim=hidden_dim,
                         num_layers=hidden_dim,
                         num_of_exm_per_length=20000,
                         word_training_length=len(dfa.states) + 10)
        print(benchmarks)
        if dir_save is not None:
            lstm.save_lstm(dir_save + "/" + "l-{}__h-{}".format(num_layers, hidden_dim))


def run_multiple_spec_on_ltsm(ltsm, spec_dfas, messages):
    i = 1
    benchmark = {}
    check_rnn_acc_to_spec(ltsm, [DFAChecker(spec_dfas[5], is_super_set=False)], benchmark,
                          timeout=1800)

    for dfa, message in zip(spec_dfas, messages):
        print(message)
        check_rnn_acc_to_spec(ltsm, [DFAChecker(dfa)], benchmark,
                              timeout=1800)
        print(benchmark)


def e_commerce_dfa():
    dfa = DFA("0", {"0,2,3,4,5"},
              {"0": {"os": "2", "gAP": "4", "gSC": "1", "bPSC": "1", "ds": "1", "eSC": "1", "aPSC": "1"},
               "1": {"os": "1", "gAP": "1", "gSC": "1", "bPSC": "1", "ds": "1", "eSC": "1", "aPSC": "1"},
               "2": {"os": "2", "gAP": "3", "gSC": "2", "bPSC": "1", "ds": "0", "eSC": "2", "aPSC": "1"},
               "3": {"os": "3", "gAP": "3", "gSC": "3", "bPSC": "1", "ds": "4", "eSC": "3", "aPSC": "5"},
               "4": {"os": "3", "gAP": "4", "gSC": "1", "bPSC": "1", "ds": "1", "eSC": "1", "aPSC": "1"},
               "5": {"os": "3", "gAP": "5", "gSC": "5", "bPSC": "3", "ds": "4", "eSC": "3", "aPSC": "5"}})
    return dfa


def alternating_bit_dfa():
    dfa = DFA("s0r1", {"s0r1"}, {"s0r1": {"msg0": "s0r0", "msg1": "sink", "ack0": "sink", "ack1": "s0r1"},
                                 "s0r0": {"msg0": "s0r0", "msg1": "sink", "ack0": "s1r0", "ack1": "sink"},
                                 "s1r0": {"msg0": "sink", "msg1": "s1r1", "ack0": "s1r0", "ack1": "sink"},
                                 "s1r1": {"msg0": "sink", "msg1": "s1r1", "ack0": "sink", "ack1": "s0r1"},
                                 "sink": {"msg0": "sink", "msg1": "sink", "ack0": "sink", "ack1": "sink"}})
    return dfa


def balanced_parentheses(word):
    count = 0
    for letter in word:
        if letter == '(':
            count += 1
        elif letter == ')':
            count -= 1
            if count < 0:
                return False
    return count == 0


def target_from_tuple(target, word):
    return target(''.join(word))


def tomita2():
    dfa = DFA("s1", {"s1"}, {"s1": {"0": "s3", "1": "s2"},
                             "s2": {"0": "s1", "1": "s3"},
                             "S3": {"0": "s3", "1": "s3"}})
    return dfa


def tomita4():
    dfa = DFA("s1", {"s1,s2,s3"}, {"s1": {"0": "s2", "1": "s1"},
                                   "s2": {"0": "s3", "1": "s1"},
                                   "S3": {"0": "s4", "1": "s1"},
                                   "S4": {"0": "s4", "1": "s4"}})
    return dfa


def tomita5():
    dfa = DFA("s1", {"s1,s2,s4"}, {"s1": {"0": "s2", "1": "s4"},
                                   "s2": {"0": "s3", "1": "s2"},
                                   "S3": {"0": "s3", "1": "s2"},
                                   "S4": {"0": "s4", "1": "s5"},
                                   "S5": {"0": "s4", "1": "s5"}})
    return dfa


def tomita6(word):
    ones = word.count('1')
    zeroes = word.count('0')
    return (zeroes - ones % 3) == 0


def run_specific_benchmarks():
    dir_name = "../models/specific/models/"
    summary_csv = "../models/specific/summary.csv"
    alphabet = ('0', '1')

    benchmark = specific_lan_benchmark(alphabet, dir_name + "tomita_1", "tomita_1", partial(target_from_tuple,
                                                                                            tomita_1))
    write_csv_header("../models/specific/summary.csv", benchmark.keys())
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    benchmark = specific_lan_benchmark(alphabet, dir_name + "tomita_2", "tomita_2", tomita2().is_word_in)
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    benchmark = specific_lan_benchmark(alphabet, dir_name + "tomita_3", "tomita_3", partial(target_from_tuple,
                                                                                            tomita_3))
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    benchmark = specific_lan_benchmark(alphabet, dir_name + "tomita_4", "tomita_4", partial(target_from_tuple,
                                                                                            tomita_4))
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    benchmark = specific_lan_benchmark(alphabet, dir_name + "tomita_5", "tomita_5", partial(target_from_tuple,
                                                                                            tomita_5))
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    benchmark = specific_lan_benchmark(alphabet, dir_name + "tomita_6", "tomita_6", partial(target_from_tuple,
                                                                                            tomita_6))
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    benchmark = specific_lan_benchmark(alphabet, dir_name + "tomita_7", "tomita_7", partial(target_from_tuple,
                                                                                            tomita_7))
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    dfa = alternating_bit_dfa()
    benchmark = specific_lan_benchmark(dfa.alphabet, dir_name + "alternating_bit_dfa", "alternating_bit_dfa",
                                       dfa.is_word_in)
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    dfa = e_commerce_dfa()
    benchmark = specific_lan_benchmark(dfa.alphabet, dir_name + "e_commerce_dfa", "e_commerce_dfa",
                                       dfa.is_word_in)
    write_line_csv(summary_csv, benchmark, benchmark.keys())

    alphabets = '()abcdefghijklmnopqrstuvwxyz'
    for i in range(6):
        alphabet = alphabets[0:2 + i * 5]
        benchmark = specific_lan_benchmark(alphabet, dir_name + "balanced_parentheses_" + str(len(alphabet - 2)),
                                           "balanced_parentheses" + str(len(alphabet - 2)), balanced_parentheses)
        write_line_csv(summary_csv, benchmark, benchmark.keys())


def specific_lan_benchmark(alphabet, dir_name, name, target):
    benchmark = {"name": name,
                 "alphabet": len(alphabet)}

    # learn target + extract
    rnn = learn_target(target, alphabet, benchmark, epoch=3, num_of_examples=2000)
    dfa_extracted = extract_dfa_from_rnn(rnn, benchmark)

    # save models:
    rnn.save_lstm(dir_name)
    save_dfa_as_part_of_model(dir_name, dfa_extracted, name=name)
    if len(dfa_extracted.states) < 50:
        dfa_extracted.draw_nicely(name="dfa_extracted_figure", save_dir=dir_name)

    # distance:
    models = [Lang(target, alphabet), rnn, dfa_extracted]
    rnn.num_of_membership_queries = 0
    compute_distances_no_model_checking(models, benchmark, delta=0.005, epsilon=0.005)

    benchmark.update({"membership_queries_distance": rnn.num_of_membership_queries})

    return benchmark


def model_check_tomita():
    dir = "../models/specific/models/"
    summary_csv = "../models/specific/model_checking_summary.csv"
    timeout = 900

    ############tomita 1#################

    rnn = RNNLanguageClasifier().load_lstm(dir + "tomita_1")
    specs = tomita_1_check_languages()
    i = 0
    for spec in specs:
        benchmark = {"name": "tomita1_" + str(i)}
        check_rnn_acc_to_spec(rnn, [DFAChecker(spec)], benchmark, timeout)
        if i == 0:
            write_csv_header(summary_csv, benchmark.keys())
        write_line_csv(summary_csv, benchmark, benchmark.keys())
        i += 1

    ############tomita 2#################

    rnn = RNNLanguageClasifier().load_lstm(dir + "tomita_2")
    specs = tomita_2_check_languages()
    i = 0
    for spec in specs:
        benchmark = {"name": "tomita2_" + str(i)}
        check_rnn_acc_to_spec(rnn, [DFAChecker(spec)], benchmark, timeout)
        write_line_csv(summary_csv, benchmark, benchmark.keys())
        i += 1

    ############tomita 3#################

    rnn = RNNLanguageClasifier().load_lstm(dir + "tomita_3")
    specs = tomita_3_check_languages()
    i = 0
    for spec in specs:
        benchmark = {"name": "tomita3_" + str(i)}
        check_rnn_acc_to_spec(rnn, [DFAChecker(spec)], benchmark, timeout)
        write_line_csv(summary_csv, benchmark, benchmark.keys())
        i += 1

    ############tomita 4#################

    rnn = RNNLanguageClasifier().load_lstm(dir + "tomita_4")
    specs = tomita_4_check_languages()
    i = 0
    for spec in specs:
        benchmark = {"name": "tomita4_" + str(i)}
        check_rnn_acc_to_spec(rnn, [DFAChecker(spec)], benchmark, timeout)
        write_line_csv(summary_csv, benchmark, benchmark.keys())
        i += 1

    # ############tomita 5#################
    #
    # rnn = RNNLanguageClasifier().load_lstm(dir + "tomita_5")
    # specs = tomita_5_check_languages()
    # i = 0
    # for spec in specs:
    #     benchmark = {"name": "tomita5_" + str(i)}
    #     check_rnn_acc_to_spec(rnn, [DFAChecker(spec)], benchmark, timeout)
    #     write_line_csv(summary_csv, benchmark, benchmark.keys)
    #     i += 1
    #
    # ###########tomita 6#################
    #
    # rnn = RNNLanguageClasifier().load_lstm(dir + "tomita_6")
    # specs = tomita_6_check_languages()
    # i = 0
    # for spec in specs:
    #     benchmark = {"name": "tomita6_" + str(i)}
    #     check_rnn_acc_to_spec(rnn, [DFAChecker(spec)], benchmark, timeout)
    #     write_line_csv(summary_csv, benchmark, benchmark.keys)
    #     i += 1

    ###########tomita 7#################

    rnn = RNNLanguageClasifier().load_lstm(dir + "tomita_7")
    specs = tomita_7_check_languages()
    i = 0
    for spec in specs:
        benchmark = {"name": "tomita7_" + str(i)}
        check_rnn_acc_to_spec(rnn, [DFAChecker(spec)], benchmark, timeout)
        write_line_csv(summary_csv, benchmark, benchmark.keys())
        i += 1


def check_folder_of_rand(folder):
    timeout = 900
    first_entry = True
    summary_csv = folder + "/summary_model_checking.csv"
    for folder in os.walk(folder):
        if os.path.isfile(folder[0] + "/meta"):
            name = folder[0].split('/')[-1]
            rnn = RNNLanguageClasifier().load_lstm(folder[0])
            dfa = load_dfa_dot(folder[0] + "/dfa.dot")
            i = 1
            for dfa_spec in from_dfa_to_sup_dfa_gen(dfa):
                dfa_spec.save(folder[0] + "/spec_" + str(i))
                benchmark = {"name": name, "spec_num": str(i)}
                check_rnn_acc_to_spec(rnn, [DFAChecker(dfa_spec)], benchmark, timeout)
                if first_entry:
                    write_csv_header(summary_csv, benchmark.keys())
                    first_entry = False
                write_line_csv(summary_csv, benchmark, benchmark.keys())
                i += 1

            # print(dfa.final_states)
            # print(dfa)

            # specs = tomita_1_check_languages()
            # i = 0
            # for spec in specs:
            #     benchmark = {"name": "tomita1_" + str(i)}
            #     check_rnn_acc_to_spec(rnn, [DFAChecker(spec)], benchmark, timeout)
            #     if i == 0:
            #         write_csv_header(summary_csv, benchmark.keys())
            #     write_line_csv(summary_csv, benchmark, benchmark.keys())
            #     i += 1
            #


def from_dfa_to_sup_dfa_gen(dfa: DFA, tries=5):
    not_final_states = [state for state in dfa.states if state not in dfa.final_states]
    if len(not_final_states) == 1:
        return

    created_dfas = []
    for _ in range(tries):
        s = np.random.randint(1, len(not_final_states))
        new_final_num = np.random.choice(len(not_final_states), size=s, replace=False)
        new_final = [not_final_states[i] for i in new_final_num]
        dfa_spec = DFA(dfa.init_state, dfa.final_states + new_final, dfa.transitions)
        dfa_spec = minimize_dfa(dfa_spec)

        if dfa_spec in created_dfas:
            continue
        created_dfas.append(dfa_spec)
        yield dfa_spec


def complition(folder):
    timeout = 900
    first_entry = True
    summary_csv = folder + "/summary_model_checking_complete.csv"
    for folder in os.walk(folder):
        if os.path.isfile(folder[0] + "/meta"):
            name = folder[0].split('/')[-1]
            rnn = RNNLanguageClasifier().load_lstm(folder[0])
            # dfa = load_dfa_dot(folder[0] + "/dfa.dot")
            for file in os.listdir(folder[0]):
                if 'spec' in file:
                    dfa_spec = load_dfa_dot(folder[0] + "/dfa.dot")
                    benchmark = {"name": name, "spec_num": file}
                    check_rnn_acc_to_spec_only_mc(rnn, [DFAChecker(dfa_spec)], benchmark, timeout)
                    if first_entry:
                        # write_csv_header(summary_csv, benchmark.keys())
                        first_entry = False
                    # write_line_csv(summary_csv, benchmark, benchmark.keys())
                  



