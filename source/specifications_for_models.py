from dfa import DFA


class Lang:
    def __init__(self, target, alphabet):
        self.target = target
        self.alphabet = alphabet

    def is_word_in(self, word):
        return self.target(word)


def tomita_1_check_languages():
    """
    :return: The Languages:
             1. there are more then two zeroes
    """

    return [DFA("s1", {"s1", "s2"}, {"s1": {"0": "s2", "1": "s1"},
                                     "s2": {"0": "s3", "1": "s2"},
                                     "s3": {"0": "s3", "1": "s3"}})]


def tomita_2_check_languages():
    """
    :return: The Languages:
             1. the lang ends with 0
    """

    return [DFA("s1", {"si,s0"}, {"si": {"0": "s0", "1": "s1"},
                               "s0": {"0": "s0", "1": "s1"},
                               "s1": {"0": "s0", "1": "s1"}})]


def tomita_3_check_languages():
    """
    :return: The Languages:
             1. add s3 as a final state
    """

    return [DFA("s1", {"s1", "s2", "s3", "s4"}, {"s1": {"0": "s1", "1": "s2"},
                                                 "s2": {"0": "s3", "1": "s1"},
                                                 "s3": {"0": "s4", "1": "s7"},
                                                 "s4": {"0": "s3", "1": "s4"},
                                                 "s7": {"0": "s7", "1": "s7"}})]


def tomita_4_check_languages():
    """
    :return: The Languagess:
             1. No more then 4 Zeroes in a row
             2. No more then 5 Zeores in a row
    """

    return [DFA("s1", {"s1,s2,s3,s4"}, {"s1": {"0": "s2", "1": "s1"},
                                        "s2": {"0": "s3", "1": "s1"},
                                        "s3": {"0": "s4", "1": "s1"},
                                        "s4": {"0": "s5", "1": "s1"},
                                        "s5": {"0": "s5", "1": "s5"}}),
            DFA("s1", {"s1,s2,s3,s4,s5"}, {"s1": {"0": "s2", "1": "s1"},
                                           "s2": {"0": "s3", "1": "s1"},
                                           "s3": {"0": "s4", "1": "s1"},
                                           "s4": {"0": "s5", "1": "s1"},
                                           "s5": {"0": "s6", "1": "s1"},
                                           "s6": {"0": "s6", "1": "s6"}})]

#Need  to  fix:
def tomita_5_check_languages():
    """
    :return: The Languagess:
             1.zeroes_or_ones_are_not_devided_by_three
             2.zeroes_or_ones_are_not_devided_by_five
    """

    return [DFA("ini", {"s00,s10,s20,s01,s11,s21,s02,s12,s22"}, {"ini": {"0": "s10", "1": "s01"},
                                                                 "s00": {"0": "s10", "1": "s01"},
                                                                 "s10": {"0": "s20", "1": "s11"},
                                                                 "s20": {"0": "s00", "1": "s21"},
                                                                 "s01": {"0": "s11", "1": "s02"},
                                                                 "s11": {"0": "s21", "1": "s12"},
                                                                 "s21": {"0": "s01", "1": "s22"},
                                                                 "s02": {"0": "s12", "1": "s00"},
                                                                 "s12": {"0": "s22", "1": "s10"},
                                                                 "s22": {"0": "s02", "1": "s20"}}),

            DFA("s1", {"s1,s2,s3,s4,s5"}, {"s1": {"0": "s2", "1": "s1"},
                                           "s2": {"0": "s3", "1": "s1"},
                                           "s3": {"0": "s4", "1": "s1"},
                                           "s4": {"0": "s5", "1": "s1"},
                                           "s5": {"0": "s6", "1": "s1"},
                                           "s6": {"0": "s6", "1": "s6"}})]


def tomita_6_check_languages():
    """
    :return: The Languages:
             1. zeroes_minus_ones_are_not_devided_by_two(
    """

    def zeroes_minus_ones_are_not_devided_by_two(word):
        return word.count('0') - (word.count('1')) % 2 != 0

    return [Lang(zeroes_minus_ones_are_not_devided_by_two, ('0', '1'))]


def tomita_7_check_languages():
    """
    :return: The Languages:
             1. lang: 0*1*0*1*0*
    """

    return [DFA("s1", {"s1", "s2", "s3", "s4"}, {"s1": {"0": "s1", "1": "s2"},
                                                 "s2": {"0": "s3", "1": "s2"},
                                                 "s3": {"0": "s3", "1": "s4"},
                                                 "s4": {"0": "s5", "1": "s4"},
                                                 "s5": {"0": "s5", "1": "s6"},
                                                 "s6": {"0": "s6", "1": "s6"}})]
