import csv
import datetime
import os
import time
from copy import deepcopy

import numpy as np

from counter_dfa import from_dfa_to_rand_counter_dfa, CounterDFA
from dfa import DFA, random_dfa, save_dfa_as_part_of_model, DFANoisy, load_dfa_dot, \
    change_n_randomly_transition, synchronised_self_with_dfa

from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import RNNLanguageClasifier
from noisy_input_dfa import NoisyInputDFA
from pac_teacher import PACTeacher, StupidGuess
from random_words import confidence_interval_many_cython

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


def extract_dfa(dfa, benchmark, suffix="", max_eq=900, word_probability=0.001, epsilon=0.001, delta=0.001):
    teacher_pac = PACTeacher(dfa, epsilon, delta, word_probability=word_probability)
    print("Starting DFA extraction")
    ###################################################
    # Doing the model checking after a DFA extraction
    ###################################################
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    # teacher_pac.teach(student, max_eq)
    teacher_pac.teach_acc_noise_dist(student, 5)
    print(student.dfa)
    benchmark.update({"extraction_time_p-" + suffix: "{:.3}".format(time.time() - start_time)})
    benchmark.update({"extraction_loops_p-" + suffix: teacher_pac._num_equivalence_asked})
    benchmark.update({"num_of_mq_p": teacher_pac.number_of_mq})
    print("time = {}".format(time.time() - start_time))
    dfa_extract = minimize_dfa(student.dfa)
    print(dfa_extract)
    benchmark.update({"dfa_extract_states_p-" + suffix: len(dfa_extract.states),
                      "dfa_extract_final_p-" + suffix: len(dfa_extract.final_states)})

    return dfa_extract


def close_rand_counter_dfa(dfa):
    for _ in range(20):
        dfa_counter = from_dfa_to_rand_counter_dfa(dfa)
        p, _ = (confidence_interval_many_cython([dfa, dfa_counter], 0.001, 0.005))
        print(p)
        if 0.0005 < p[1][0] < 0.01:
            return dfa_counter

    return dfa_counter


def extract_mesaure(dfa: DFA, benchmark, dir_name=None, epsilons=[0.005], p_noise=[0.01, 0.005, 0.0025, 0.0015, 0.001],
                    dfa_noise=DFANoisy):
    max_eq = 250
    num_of_retry = 3
    for p in p_noise:
        print("Running p = {}:".format(p))
        if dfa_noise == CounterDFA:
            dfa_noisy = close_rand_counter_dfa(dfa)
        else:
            dfa_noisy = dfa_noise(dfa.init_state, dfa.final_states, dfa.transitions, mistake_prob=p)
        for epsilon in epsilons:
            for word_prob in [0.01]:
                models = [dfa, dfa_noisy]
                for _ in range(num_of_retry):
                    suffix = "EpDel-" + str(epsilon) + "MaxEQ" + str(max_eq) + "WProb" + str(word_prob)
                    if type(dfa_noisy) == CounterDFA:
                        suffix = "counter_dfa" + suffix
                    else:
                        suffix = str(dfa_noisy.mistake_prob) + suffix

                    extracted_dfa = extract_dfa(dfa_noisy, benchmark, suffix, max_eq, word_prob,
                                                epsilon=epsilon, delta=epsilon)

                    if dir_name is not None:
                        save_dfa_as_part_of_model(dir_name, extracted_dfa,
                                                  name="extracted_dfa_p" + suffix + "-3",
                                                  force_overwrite=True)
                    models.append(extracted_dfa)
                # models = [dfa, dfa_noisy]
                # models.extend(extracted_dfas)
                compute_distances(models, benchmark, suffix=suffix, epsilon=0.0005,
                                  word_prob=word_prob)

            # print(benchmark)


def compute_distances(models, benchmark, epsilon=0.001, delta=0.005, suffix="", word_prob=0.01):
    startime = time.time()
    print("Starting distance measuring")
    output, samples = confidence_interval_many_cython(models, width=epsilon, confidence=delta, word_prob=word_prob)
    print("The confidence interval for epsilon = {} , delta = {}".format(epsilon, delta))
    print(output)
    dist_2_original = np.average(output[0][2:])
    dist_2_noisy = np.average(output[1][2:])
    benchmark.update({"dist_dfa_vs_noisy_p-" + suffix: "{}".format(output[0][1]),
                      "dist_dfa_vs_extr_p-" + suffix: "{}".format(dist_2_original),
                      "dist_noisy_vs_extr-p-" + suffix: "{}".format(dist_2_noisy)})

    # benchmark.update({"dist_dfa_vs_rand_trans-" + suffix: "{}".format(output[0][1])})

    print(output)

    print("Finished distance measuring in {}'s".format(time.time() - startime))


def benchmarks_noise_model(num_of_bench=10, epsilons=[0.005], p_noise=[0.01, 0.005, 0.0025, 0.0015, 0.001],
                           dfa_noise=DFANoisy,
                           title4csv=False,
                           save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_noisy_dfa_{}".format(
            datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    for num in range(1, num_of_bench + 1):
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(
            save_dir + "/" + format(datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S")) + str(num)
            , epsilons, p_noise, dfa_noise)
        if num == 1 and title4csv:
            write_csv_header(save_dir + "/test.csv", benchmark.keys())
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())


def rand_benchmark(save_dir=None, epsilons=[0.005], p_noise=[0.01, 0.005, 0.0025, 0.0015, 0.001], dfa_noise=DFANoisy):
    full_alphabet = "abcdefghijklmnopqrstuvwxyz"

    alphabet = full_alphabet[0:np.random.randint(4, 20)]
    # alphabet = full_alphabet[0:19]

    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    while True:
        dfa_rand = random_dfa(alphabet, min_state=20, max_states=60)
        dfa = minimize_dfa(dfa_rand)
        if len(dfa.states) > 20:
            break

    benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

    if save_dir is not None:
        save_dfa_as_part_of_model(save_dir, dfa, name="dfa")

    print("DFA to learn {}".format(dfa))

    extract_mesaure(dfa, benchmark, save_dir, epsilons, p_noise, dfa_noise)

    return benchmark


# -----------------------------------------------------------------------
# --------------------------TO DELETE------------------------------------
# -----------------------------------------------------------------------

def rand_benchmark_enumerable(save_dir=None, dfa_noise=DFANoisy, not_re=True):
    full_alphabet = "abcdefghijklmnopqrstuvwxyz"

    # alphabet = full_alphabet[0:np.random.randint(4, 20)]
    # alphabet = full_alphabet[0:8]
    # benchmark = {}
    # benchmark.update({"alph_len": len(alphabet)})

    # max_final = np.random.randint(6, 40)
    #
    # dfa_rand = random_dfa(alphabet, min_states=max_final + 1, max_states=50, min_final=5, max_final=max_final)
    # dfa = minimize_dfa(dfa_rand)
    j = 1
    start_time = time.time()
    while True:
        if j % 100 == 0:
            print("j = {}  after {} seconds from start".format(j, time.time() - start_time))
        j = j + 1

        # alphabet = full_alphabet[0:np.random.randint(3, 6)]
        alphabet = full_alphabet[0:3]

        max_final = np.random.randint(20, 45)
        # print(max_final)
        # print(len(alphabet))
        dfa_rand = random_dfa(alphabet)
        dfa = minimize_dfa(dfa_rand)

        if len(dfa.states) < 6:
            continue

        sync_dfa = synchronised_self_with_dfa(dfa, dfa)

        bscc = sync_dfa.bottom_strongly_connected_components()

        found = False
        for scc in bscc:
            for s in scc:
                if s in sync_dfa.final_states:
                    found = True
                    continue
            if found:
                break
        if found == not_re:
            break

    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

    if save_dir is not None:
        save_dfa_as_part_of_model(save_dir, dfa, name="dfa")

    print("DFA to learn {}".format(dfa))

    extract_mesaure(dfa, benchmark, save_dir, dfa_noise)

    return benchmark


def run_benchmark(dfa: DFA, save_dir, dfa_noise):
    # full_alphabet = "abcdefghijklmnopqrstuvwxyz"
    #
    # alphabet = full_alphabet[0:np.random.randint(4, 20)]
    # # alphabet = full_alphabet[0:8]
    benchmark = {}
    benchmark.update({"alph_len": len(dfa.alphabet)})
    #
    # max_final = np.random.randint(6, 40)
    #
    # dfa_rand = random_dfa(alphabet, min_states=max_final + 1, max_states=50, min_final=5, max_final=max_final)
    # dfa = minimize_dfa(dfa_rand)

    benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

    # if save_dir is not None:
    #     save_dfa_as_part_of_model(save_dir, dfa, name="dfa")

    print("DFA to learn {}".format(dfa))

    extract_mesaure(dfa, benchmark, save_dir, dfa_noise)

    return benchmark


def run_rand_benchmarks_noisy_dfa(num_of_bench=10, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_noisy_dfa_{}".format(
            datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    for num in range(1, num_of_bench + 1):
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num))
        if num == 1:
            write_csv_header(save_dir + "/test.csv", benchmark.keys())
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())


def run_rand_benchmarks_noisy_input_dfa2(num_of_bench=10, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_noisy_dfa_{}".format(
            datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    for num in range(1, num_of_bench + 1):
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num), dfa_noise=NoisyInputDFA)
        if num == 1:
            write_csv_header(save_dir + "/test.csv", benchmark.keys())
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())


def number_of_rounds_analisys(num_of_bench=10, p=0.001, save_dir_main=None):
    if save_dir_main is None:
        save_dir_main = "../models/random_bench_noisy_dfa_num_rounds_{}".format(
            datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir_main)

    for num in range(1, num_of_bench + 1):
        save_dir = save_dir_main + "/" + str(num)
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        full_alphabet = "abcdefghijklmnopqrstuvwxyz"

        alphabet = full_alphabet[0:np.random.randint(4, 20)]
        benchmark = {}
        benchmark.update({"alph_len": len(alphabet)})

        max_final = np.random.randint(15, 50)

        dfa_rand = random_dfa(alphabet, max_states=50)
        dfa = minimize_dfa(dfa_rand)

        benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

        if save_dir is not None:
            save_dfa_as_part_of_model(save_dir, dfa, name="dfa")

        print("DFA to learn {}".format(dfa))

        for p in [p]:
            dfa_noisy = DFANoisy(dfa.init_state, dfa.final_states, dfa.transitions, mistake_prob=p)
            dist = extract_for_masure(benchmark, dfa, dfa_noisy, str(dfa_noisy.mistake_prob), save_dir)

        if num == 1:
            write_csv_header(save_dir + "/test.csv", benchmark.keys())
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())
        for i in range(len(dist[0])):
            row = {"dfa_vs_extracted_p-" + str(dfa_noisy.mistake_prob): dist[0][i],
                   "dfa_vs_noisy_p-" + str(dfa_noisy.mistake_prob): dist[1][i]}
            write_line_csv(save_dir + "/test.csv", row, benchmark.keys())


def extract_for_masure(benchmark, dfa, dfa_noisy, sufix, save_dir):
    teacher_pac = PACTeacher(dfa_noisy)
    print("Starting DFA extraction")
    ###################################################
    # Doing the model checking after a DFA extraction
    ###################################################
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    # dist_to_dfa_vs, dist_to_rnn_vs, num_of_states
    disttances = teacher_pac.teach_and_trace(student, dfa, timeout=600)
    benchmark.update(
        {"dfa_vs_extracted_p-" + sufix: 0,
         "dfa_vs_noisy_p-" + sufix: 0,
         "number of querries_p-" + sufix: 0})
    print(disttances)
    benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})
    benchmark.update({"extraction_loops": teacher_pac._num_equivalence_asked})
    print("time = {}".format(time.time() - start_time))
    dfa_extract = minimize_dfa(student.dfa)
    print(student.dfa)
    benchmark.update({"dfa_extract_states_p-" + sufix: len(dfa_extract.states),
                      "dfa_extract_final_p-" + sufix: len(dfa_extract.final_states)})
    if save_dir is not None:
        save_dfa_as_part_of_model(save_dir, dfa_extract, name="extracted_dfa_p" + sufix)
    return disttances
    # models = [dfa, dfa_noisy, dfa_extract]
    # if isinstance(dfa_noisy, DFANoisy):
    #     compute_distances(models, benchmark)


def remasure_dfa(dir):
    first_entry = True
    summary_csv = dir + "/p001-epsilon-00001.csv"
    for folder in os.walk(dir):
        if os.path.isfile(folder[0] + "/dfa.dot"):
            dfa = load_dfa_dot(folder[0] + "/dfa.dot")
            # print(dfa)
            # benchmark = {"name": name, "spec_num": file}
            benchmark = {"alph_len": len(dfa.alphabet)}
            benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})
            extract_mesaure(dfa, benchmark, folder[0])
            print(benchmark)
            if first_entry:
                write_csv_header(summary_csv, benchmark.keys())
                first_entry = False
            write_line_csv(summary_csv, benchmark, benchmark.keys())
    # for num in range(1, num_of_bench + 1):
    #
    #     print("Running benchmark {}/{}:".format(num, num_of_bench))
    #     benchmark = rand_benchmark(save_dir + "/" + str(num))
    #     if num == 1:
    #         write_csv_header(save_dir + "/test.csv", benchmark.keys())
    #     print("Summary for the {}th benchmark".format(num))
    #     print(benchmark)
    #     write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())


def run_rand_benchmarks_noisy_input_dfa(num_of_bench=10, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_bench_area3x2_change{}".format(
            datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    for num in range(1, num_of_bench + 1):
        start_time = time.time()
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num), NoisyInputDFA)
        if num == 1:
            write_csv_header(save_dir + "/test.csv", benchmark.keys())
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())
        print("time for bench mark {}".format(time.time() - start_time))


def run_rand_benchmarks_noisy_input_dfa_enumerable(num_of_bench=10, save_dir=None, not_re=True):
    if save_dir is None:
        save_dir = "../models/random_input_3_letters_not_re={}_{}".format(not_re,
                                                                          datetime.datetime.now().strftime(
                                                                              "%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    for num in range(1, num_of_bench + 1):
        start_time = time.time()
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark_enumerable(save_dir + "/" + str(num), NoisyInputDFA, not_re=not_re)
        if num == 1:
            write_csv_header(save_dir + "/test.csv", benchmark.keys())
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())
        print("time for bench mark {}".format(time.time() - start_time))


def run_rand_benchmarks_noisy_input_dfa_on_given_dit(dir):
    save_dir = dir + "/results"
    # os.makedirs(save_dir)
    num = 1
    for folder in os.walk(dir):
        # print(folder)
        for f in folder[2]:
            print(f)
            if ".dot" in f:
                dfa = load_dfa_dot(dir + "/" + f)
            else:
                continue

            start_time = time.time()
            print("Running benchmark {}:".format(f))
            benchmark = run_benchmark(dfa, save_dir + "/" + str(num), NoisyInputDFA)
            if num == 1:
                write_csv_header(save_dir + "/test.csv", benchmark.keys())
            print("Summary for the {}th benchmark".format(num))
            print(benchmark)
            write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())
            print("time for bench mark {}".format(time.time() - start_time))
            num += 1


def run_rand_benchmarks_noisy_input_dfa(num_of_bench=50, save_dir=None):
    if save_dir is None:
        save_dir = "../models/random_trans_change_{}".format(
            datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)
    first = True
    res = []
    for states in range(1, 10):
        for num_final in range(2, 5):
            for num in range(1, num_of_bench + 1):
                start_time = time.time()
                print("Running benchmark {}/{}:".format(num, num_of_bench))
                benchmark = {}

                full_alphabet = "abcdefghijklmnopqrstuvwxyz"
                alphabet = full_alphabet[0: 10]
                benchmark.update({"alph_len": len(alphabet)})

                # max_final = np.random.randint(6, 40)
                while True:
                    min_states = states * 10 + 5
                    dfa_rand = random_dfa(alphabet)
                    dfa = minimize_dfa(dfa_rand)
                    if states * 10 - 5 < len(dfa.states) < states * 10 + 5:
                        break
                    print(states)
                    print(len(dfa.states))
                    print("nope")

                benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

                dfarand = change_n_randomly_transition(dfa, 1)
                compute_distances([dfa, dfarand], benchmark, suffix="tran_changed_{}_".format(1), epsilon=0.005)

                if first:
                    print(benchmark)
                    write_csv_header(save_dir + "/test.csv", benchmark.keys())
                    first = False
                print("Summary for the {}th benchmark".format(num))
                print(benchmark)
                write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())
                res.append(deepcopy(benchmark));
                print("time for bench mark {}".format(time.time() - start_time))
            for key in benchmark.keys():
                benchmark[key] = "*"
            write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())
        for key in benchmark.keys():
            benchmark[key] = "-"
        write_line_csv(save_dir + "/test.csv", benchmark, benchmark.keys())
    return res
