import numpy as np

from benchmarking_counter_dfa import run_rand_benchmarks_counter_dfa
from benchmarking_noisy_dfa import run_rand_benchmarks_noisy_dfa, number_of_rounds_analisys, \
    run_rand_benchmarks_noisy_input_dfa2, benchmarks_noise_model
from counter_dfa import CounterDFA
from dfa import DFA, random_dfa, synchronised_self_with_dfa
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from noisy_input_dfa import NoisyInputDFA

benchmarks_noise_model(num_of_bench=1, p_noise=[0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001], title4csv=True,
                       save_dir=None)
benchmarks_noise_model(num_of_bench=1, p_noise=[0.0025, 0.002, 0.001, 0.0075, 0.0005], dfa_noise=NoisyInputDFA,
                       title4csv=True, save_dir=None)
benchmarks_noise_model(num_of_bench=5, p_noise=[1], title4csv=True, dfa_noise=CounterDFA, save_dir=None)

quit()
