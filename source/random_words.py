import sys
import time
import numpy as np
from randwords import random_words, is_words_in_dfa, compare_list_of_bool, is_words_in_counterDfa, \
    is_words_in_dfa_finalcount

from counter_dfa import CounterDFA, NoisyCounterDFA, DFAFinalCount
from dfa import DFA, DFANoisy

from noisy_input_dfa import NoisyInputDFA


def random_word(alphabet, p=0.01):
    nums_of_letters = len(alphabet)
    word = []
    while np.random.randint(0, int(1 / p)) != 0:
        letter = np.random.randint(0, nums_of_letters)
        word.append(alphabet[letter])
    return tuple(word)


def random_word_by_letter(alphabet, p=0.01):
    nums_of_letters = len(alphabet)
    while np.random.randint(0, int(1 / p)) != 0:
        letter = np.random.randint(0, nums_of_letters)
        yield alphabet[letter]


def confidence_interval(language1, language2, sampler, delta=0.001, epsilon=0.001, samples=None):
    n = np.log(2 / delta) / (2 * epsilon * epsilon)
    print(n)
    if samples is None:
        samples = set()
        while len(samples) < n:
            w = sampler(language1.alphabet)
            if w not in samples:
                samples.add(w)
            # print(len(samples))
    mistakes = 0
    print("got it")
    for w in samples:
        if language1.is_word_in(w) != language2.is_word_in(w):
            mistakes = mistakes + 1
            # print(mistakes)
    return mistakes / n, samples


def confidence_interval_many(languages, sampler, confidence=0.001, width=0.001, samples=None):
    """
    Produce the probabilistic distance of the given languages. Using the Chernoff-Hoeffding bound we get that
    in order to have:
        P(S - E[S]>width)< confidence
        S = 1/n(n empirical examples)

    the number of examples that one needs to use is:
        #examples = log(2 / confidence) / (2 * width * width)

    For more details:
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    """
    num_of_lan = len(languages)
    if num_of_lan < 2:
        raise Exception("Need at least 2 languages to compare")

    num_of_samples = np.log(2 / confidence) / (2 * width * width)
    print("size of sample:" + str(int(num_of_samples)))
    if samples is None:
        samples = [sampler(languages[0].alphabet) for _ in range(int(num_of_samples))]

    in_langs_lists = []
    i = 0
    sys.stdout.write('\r Creating bool lists for each lan:  {}/{} done'.format(i, num_of_lan))
    for lang in languages:
        tmp = []
        for w in samples:
            tmp.append(lang.is_word_in(w))
        in_langs_lists.append(tmp)

    output = []
    for i in range(num_of_lan):
        output.append([1] * num_of_lan)

    for lang1 in range(num_of_lan):
        for lang2 in range(num_of_lan):
            if lang1 == lang2:
                output[lang1][lang2] = 0
            elif output[lang1][lang2] == 1:
                output[lang1][lang2] = ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
                                         range(len(samples))].count(False)) / num_of_samples

    print()
    return output, samples


def confidence_interval_subset(language_inf, language_sup, samples=None, confidence=0.001, width=0.001):
    """
    Getting the confidence interval(width,confidence) using the Chernoff-Hoeffding bound.
    The number of examples that one needs to use is n= log(2 / confidence) / (2 * width * width.
    For more details:
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    :return:
    """
    start_time = time.time()
    n = np.log(2 / confidence) / (2 * width * width)

    if samples is None:
        samples = []
        while len(samples) <= n:
            # if len(samples) % 1000 == 0:
            #     sys.stdout.write('\r Creating words:  {}/100 done'.format(str(int((len(samples) / n) * 100))))
            samples.append(random_word(language_inf.alphabet))

        sys.stdout.write('\r Creating words:  100/100 done \n')

    mistakes = 0

    for w in samples:
        if (language_inf.is_word_in(w)) and (not language_sup.is_word_in(w)):
            if mistakes == 0:
                print("first mistake")
                print(time.time() - start_time)
            mistakes = mistakes + 1
    return mistakes / n, samples


def confidence_interval_many_for_reuse(languages, sampler, previous_answers=None, confidence=0.001, width=0.005,
                                       samples=None):
    """
    Produce the probabilistic distance of the given languages. Using the Chernoff-Hoeffding bound we get that
    in order to have:
        P(S - E[S]>width)< confidence
        S = 1/n(n empirical examples)

    the number of examples that one needs to use is:
        #examples = log(2 / confidence) / (2 * width * width)

    For more details:
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    """
    st = time.time()
    num_of_lan = len(languages)
    if num_of_lan < 2:
        raise Exception("Need at least 2 languages to compare")

    n = np.log(2 / confidence) / (2 * width * width)
    # print("size of sample:" + str(int(n)))
    if samples is None:
        samples = []

        # for i in range(5):
        #     random_words_c(3800451, ('a', 'b', 'c', 'd', 'e'), 100)
        #     samples.extend(None)
        # print(n)
        # print(len(samples))
        print("begin samples {} creation for width ={} and confidence = {}".format(n, width, confidence))
        samples = random_words(n, tuple(languages[0].alphabet), 1 / 0.01)

        # while len(samples) <= n:
        #     if len(samples) % 1000 == 0:
        #         sys.stdout.write('\r Creating words:  {}/100 done'.format(str(int((len(samples) / n) * 100))))
        #     samples.append(sampler(languages[0].alphabet))

        sys.stdout.write('\r Creating words:  100/100 done \n')
    in_langs_lists = []
    # i = 0
    # sys.stdout.write('\r Creating bool lists for each lan:  {}/{} done'.format(i, num_of_lan))
    if previous_answers is None:
        for lang in languages:
            in_langs_lists.append([lang.is_word_in(w) for w in samples])
    else:
        in_langs_lists = previous_answers
        in_langs_lists.append([languages[-1].is_word_in(w) for w in samples])
        # print(in_langs_lists)
    output = []
    for _ in range(num_of_lan):
        output.append([1] * num_of_lan)

    # for lang1 in range(num_of_lan):
    #     for lang2 in range(num_of_lan):
    #         if lang1 == lang2:
    #             output[lang1][lang2] = 0
    #         elif output[lang1][lang2] == 1:
    #             output[lang1][lang2] = ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
    #                                      range(len(samples))].count(False)) / len(samples)
    print("finished in {} s".format(time.time() - st))

    # print()
    output[0][1] = ([(in_langs_lists[0])[i] == (in_langs_lists[2])[i] for i in
                     range(len(samples))].count(False)) / len(samples)
    output[0][2] = ([(in_langs_lists[1])[i] == (in_langs_lists[2])[i] for i in
                     range(len(samples))].count(False)) / len(samples)
    return output, samples, in_langs_lists[0:-1]


def model_check_random(language_inf, language_sup, confidence=0.001, width=0.001):
    """


    :return:
    """
    n = np.log(2 / confidence) / (2 * width * width)

    batch_size = 200
    for i in range(int(n / batch_size) + 1):
        batch = [random_word(language_inf.alphabet) for _ in range(batch_size)]
        for x, y, w in zip(language_inf.is_words_in_batch(batch) > 0.5, [language_sup.is_word_in(w) for w in batch],
                           batch):
            if x and (not y):
                return w
    return None


def confidence_interval_many_cython(languages, confidence=0.001, width=0.005, samples=None, word_prob=0.01):
    """
    Produce the probabilistic distance of the given languages. Using the Chernoff-Hoeffding bound we get that
    in order to have:
        P(S - E[S]>width)< confidence
        S = 1/n(n empirical examples)

    the number of examples that one needs to use is:
        #examples = log(2 / confidence) / (2 * width * width)

    For more details:
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    """
    num_of_lan = len(languages)
    if num_of_lan < 2:
        raise Exception("Need at least 2 languages to compare")

    full_num_of_samples = np.log(2 / confidence) / (2 * width * width)
    max_samples = 1000000
    output = []
    # num_of_lan = 3
    for i in range(num_of_lan):
        output.append([0] * num_of_lan)
    div = int(full_num_of_samples / max_samples)
    mod = full_num_of_samples % max_samples
    if div >= 1:
        nums = [max_samples] * div
        nums.append(mod)
    else:
        nums = [mod]
    # print(div, mod, nums, full_num_of_samples)

    runs_done = 0
    samples = None
    for num_of_samples in nums:
        # time.sleep(30)
        # if runs_done % 5 == 0:
        #     print("process {}/{}:".format(runs_done, len(nums)))
        runs_done += 1

        del samples
        samples = None

        # if samples is None:
        # print(int(1 / word_prob))
        def sampler():
            return random_words(num_of_samples, tuple(languages[0].alphabet), 1 / word_prob)

        samples = sampler()

        # samples = [random_word(tuple(languages[0].alphabet)) for _ in range(max_samples)]
        # print(len(samples))
        # if len(samples1) != 0:
        #     for w in samples1:
        #         if w in samples:
        #             print("yeah")
        #
        # samples1 = samples
        # samples = [sampler(languages[0].alphabet) for _ in range(int(num_of_samples))]
        #
        # print(word_prob)
        # print(len(samples))
        # print(sum([len(w) for w in samples])/len(samples))
        in_langs_lists = []
        i = 0
        # sys.stdout.write('\r Creating bool lists for each lan:  {}/{} done'.format(i, num_of_lan))
        # torch.cuda.empty_cache() /notsure wtf is this??
        for lang in languages:
            # print(lang)
            if isinstance(lang, DFANoisy) or isinstance(lang, NoisyInputDFA) or isinstance(lang, NoisyCounterDFA):
                # print("noisy DFA")
                in_langs_lists.append([lang.is_word_in(w) for w in samples])
            elif isinstance(lang, DFAFinalCount):
                in_langs_lists.append(is_words_in_dfa_finalcount(lang, samples))
            elif isinstance(lang, CounterDFA):
                in_langs_lists.append(is_words_in_counterDfa(lang, samples))
                # in_langs_lists.append([lang.is_word_in(w) for w in samples])
            elif not isinstance(lang, DFA):
                in_langs_lists.append([lang.is_word(w) for w in samples])
            else:
                # print("")
                in_langs_lists.append(is_words_in_dfa(lang, samples))
            # in_langs_lists.append([lang.is_word_in(w) for w in samples])
            # print(in_langs_lists)

        for lang1 in range(num_of_lan):
            for lang2 in range(num_of_lan):
                if lang1 == lang2:
                    output[lang1][lang2] = 0
                else:
                    # output[lang1][lang2] = compare_list_of_bool(in_langs_lists[lang1], in_langs_lists[lang2],
                    #                                             int(num_of_samples))
                    output[lang1][lang2] += ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
                                              range(len(samples))].count(False))
        # for lang1 in range(num_of_lan):
        #     in_langs_lists[lang1] = []

    for lang1 in range(num_of_lan):
        for lang2 in range(num_of_lan):
            # print(output[lang1][lang2])
            output[lang1][lang2] = output[lang1][lang2] / full_num_of_samples
    # print("done with confidence interval")
    return output, samples


def random_words(batch_size, alphabet, word_length):
    return [random_word(alphabet, 1 / word_length) for _ in range(int(batch_size))]


def confidence_interval_many_for_reuse_2(languages, sampler, previous_answers=None, confidence=0.001, width=0.005,
                                         samples=None):
    """
    Produce the probabilistic distance of the given languages. Using the Chernoff-Hoeffding bound we get that
    in order to have:
        P(S - E[S]>width)< confidence
        S = 1/n(n empirical examples)

    the number of examples that one needs to use is:
        #examples = log(2 / confidence) / (2 * width * width)

    For more details:
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    """
    st = time.time()
    num_of_lan = len(languages)
    if num_of_lan < 2:
        raise Exception("Need at least 2 languages to compare")

    n = np.log(2 / confidence) / (2 * width * width)
    # print("size of sample:" + str(int(n)))
    if samples is None:
        samples = []

        # for i in range(5):
        #     random_words_c(3800451, ('a', 'b', 'c', 'd', 'e'), 100)
        #     samples.extend(None)
        # print(n)
        # print(len(samples))
        print("begin samples {} creation for width ={} and confidence = {}".format(n, width, confidence))
        samples = random_words(n, tuple(languages[0].alphabet), 1 / 0.01)

        # while len(samples) <= n:
        #     if len(samples) % 1000 == 0:
        #         sys.stdout.write('\r Creating words:  {}/100 done'.format(str(int((len(samples) / n) * 100))))
        #     samples.append(sampler(languages[0].alphabet))

        sys.stdout.write('\r Creating words:  100/100 done \n')
    in_langs_lists = []
    # i = 0
    # sys.stdout.write('\r Creating bool lists for each lan:  {}/{} done'.format(i, num_of_lan))
    if previous_answers is None:
        for lang in languages:
            in_langs_lists.append([lang.is_word_in(w) for w in samples])
    else:
        in_langs_lists = previous_answers
        in_langs_lists.append([languages[-1].is_word_in(w) for w in samples])
        # print(in_langs_lists)
    output = []
    for _ in range(num_of_lan):
        output.append([1] * num_of_lan)

    # for lang1 in range(num_of_lan):
    #     for lang2 in range(num_of_lan):
    #         if lang1 == lang2:
    #             output[lang1][lang2] = 0
    #         elif output[lang1][lang2] == 1:
    #             output[lang1][lang2] = ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
    #                                      range(len(samples))].count(False)) / len(samples)
    print("finished in {} s".format(time.time() - st))

    output[0][1] = ([(in_langs_lists[0])[i] == (in_langs_lists[1])[i] for i in
                     range(len(samples))].count(False)) / len(samples)
    return output, samples, in_langs_lists[0:-1]
