import cython
from libc.stdlib cimport rand, RAND_MAX
# from libcpp cimport bool

@cython.boundscheck(False)
#
def random_words(int num_of_words,tuple alphabet, double one_div_p=100.0):
    words = []
    cdef int nums_of_letters = len(alphabet)
    # cdef int one_div_p = int(1/p)
    # print("rand()")
    # print((RAND_MAX/one_div_p))
    cdef int letter
    cdef int a
    for i in range(num_of_words):
        word = []
        # print(a)
        a = int(rand()/(RAND_MAX/one_div_p))
        while a != 0:
            a = int(rand()/(RAND_MAX/one_div_p))
            # letter = np.random.randint(0, nums_of_letters)
            letter =  int(rand()/(RAND_MAX/nums_of_letters +1))
            # if letter >= nums_of_letters:
            #   print("problem")
            #   print(letter)
            #   print(len(alphabet))
            #   print(nums_of_letters)
            word.append(alphabet[letter])

        words.append(tuple(word))
    return words

def is_words_in_dfa(dfa, words):
    lenwords = len(words)
    outputs = []
    final_states = dfa.final_states
    init_state = dfa.init_state
    transitions = dfa.transitions
    for i in range(lenwords):
        state = init_state
        word = words[i]
        lenword= len(word)
        for j in range(lenword):
            state = transitions[state][word[j]]

        outputs.append(state in final_states)

    return outputs

def is_words_in_counterDfa(counter_dfa, words):
    lenwords = len(words)
    outputs = []
    alph_counter = counter_dfa.alphabet2counter
    final_states = counter_dfa.final_states
    init_state = counter_dfa.init_state
    transitions = counter_dfa.transitions
    cdef int init_counter = counter_dfa.init_tokens
    cdef int counter
    # sup = counter_dfa.sup
    cdef bint sup = counter_dfa.sup

    for i in range(lenwords):
        state = init_state
        counter = init_counter
        word = words[i]
        lenword= len(word)
        for j in range(lenword):
            state = transitions[state][word[j]]
            counter += alph_counter[word[j]]

        if counter<0:
            outputs.append(sup)
        else:
            outputs.append(state in final_states)
        #
        #
        # outputs.append(((state in final_states) & (counter>=0 )) | (sup & (counter<0 )))

    return outputs

def compare_list_of_bool(lang1, lang2,num_of_samples):
        # lenwords = len(lang1)
        cdef int count = 0
        for i in range(num_of_samples):
            count += (lang1[i]!=lang2[i])


        return (count/num_of_samples)


    # ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
    #                                         range(len(samples))].count(False)) / num_of_samples    # state = self.init_state
    # #         for letter in word:
    #             state = self.transitions[state][letter]
    #         return state in self.final_states


def scrumble_word(word,alphabet,double p=0.001):
  cdef int nums_of_letters = len(alphabet)
  cdef int one_div_p = int(1/p)
  cdef int letter
  cdef int len_word = len(word)
  cdef double max_change = RAND_MAX/one_div_p
  cdef double alphabet_prob = (RAND_MAX/nums_of_letters) +1
  new_word =list(word)
  for i in range(len_word):
      change = int(rand()/max_change)
      if change == 0:
          letter =  int(rand()/alphabet_prob)
          new_word[i] = alphabet[letter]
  return new_word



def scrumble_word_orderly(word,alphabet,double p=0.001):
  alphabet_turn  = {'a':'b','b':'b','c':'a','d':'e','e':'e','f':'g','g':'h','h':'a'}
  cdef int nums_of_letters = len(alphabet)
  cdef int one_div_p = int(1/p)
  cdef int letter
  cdef int len_word = len(word)
  cdef double max_change = RAND_MAX/one_div_p
  cdef double alphabet_prob = (RAND_MAX/nums_of_letters) +1
  new_word =list(word)
  for i in range(len_word):
      change = int(rand()/max_change)
      if change == 0:
          new_word[i] = alphabet_turn[new_word[i]]
  return new_word


def scrumble_word_reducing(word,alphabet,double p=0.001,double dec_p = 1.1):
  cdef int nums_of_letters = len(alphabet)
  cdef int one_div_p = int(1/p)
  cdef int letter,i
  cdef int len_word = len(word)
  cdef double max_change = RAND_MAX/one_div_p
  cdef double alphabet_prob = (RAND_MAX/nums_of_letters) + 1

  new_word =list(word)

  for i in range(len_word):
      # change = int(rand()/max_change)
      if int(rand()/max_change) == 0:
          letter =  int(rand()/alphabet_prob)
          new_word[len_word-i-1] = alphabet[letter]
      max_change = max_change/dec_p
      # print(max_change)
  if int(rand()/(RAND_MAX/(one_div_p/2))) == 0:
    letter =  int(rand()/alphabet_prob)
    new_word.append(alphabet[letter])
  return tuple(new_word)


def sto_is_word_in(dfa , word , p):
  transitions = dfa.transitions
  states = dfa.states
  len_state = len(states)
  cdef int one_div_p = int(1/p)
  cdef double mistak_const = RAND_MAX/one_div_p
  cdef double state_const = RAND_MAX/len_state
  state = dfa.init_state
  lenword= len(word)
  for j in range(lenword):
      a = int(rand()/(mistak_const))
      if a == 0:
          state = states[int(rand()/(state_const))]
      else:
          state = transitions[state][word[j]]

  return state in dfa.final_states


def is_words_in_dfa_finalcount(dfa_finalcount, words):
    lenwords = len(words)
    outputs = []
    final_states = dfa_finalcount.final_states
    init_state = dfa_finalcount.init_state
    transitions = dfa_finalcount.transitions
    states_close_to_final = dfa_finalcount.states_close_to_final

    cdef double count_ini = dfa_finalcount.init_count
    cdef double multi = dfa_finalcount.multiplier
    cdef double threshold = dfa_finalcount.threshold
    cdef double second_add = dfa_finalcount.second_add
    cdef long double count


    for i in range(lenwords):
        state = init_state
        count = count_ini
        word = words[i]
        lenword = len(word)
        #if lenword == 0:
        #    outputs.append(init_state in final_states)
        #    continue
        for j in range(lenword):
            count *= multi
            state = transitions[state][word[j]]
            count += (state in final_states)
            count += second_add * (state in states_close_to_final)

        outputs.append(count > threshold)

    return outputs
