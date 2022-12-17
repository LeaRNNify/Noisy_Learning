import logging

import numpy as np

from benchmarking_noisy_dfa import BenchmarkingNoise
from counter_dfa import CounterDFA
from noisy_input_dfa import NoisyInputDFA

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

np.random.seed(seed=2)
# todo remove epsilon and 0.001 noise
benchmark_noisy_dfa = BenchmarkingNoise(epsilons=(0.001,), p_noise=[0.01, 0.005, 0.0025, 0.0015, 0.001])
benchmark_noisy_dfa.benchmarks_noise_model(10)
benchmark_noisy_dfa = BenchmarkingNoise(epsilons=(0.001,), p_noise=[0.005, 0.001, 0.0005, 0.0001], dfa_noise=NoisyInputDFA)
benchmark_noisy_dfa.benchmarks_noise_model(10)
benchmark_noisy_dfa = BenchmarkingNoise(epsilons=(0.001,), p_noise=[], dfa_noise=CounterDFA)
benchmark_noisy_dfa.benchmarks_noise_model(100)

# benchmarks_noise_model(num_of_bench=2, epsilons=(0.1,), p_noise=[0.005, 0.001], save_dir=None)
# benchmarks_noise_model(num_of_bench=1, p_noise=[0.0005], dfa_noise=NoisyInputDFA,
#                        title4csv=True, save_dir=None)
# benchmarks_noise_model(num_of_bench=1, p_noise=[1], title4csv=True, dfa_noise=CounterDFA, save_dir=None)

quit()
