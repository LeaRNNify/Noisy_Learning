from graphviz import Digraph

from dfa import DFA
from learner import Learner


class TreeNode:
    def __init__(self, name=tuple(), in_lan=True, parent=None):
        self.right = None
        self.left = None
        self.name = name
        self.inLan = in_lan
        self.parent = parent

        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def __repr__(self):
        return "TreeNode: name: \'" + self.name + "\', depth:" + str(self.depth)

    def draw(self, filename):
        graph = Digraph('G', filename=filename)
        front = []

        if self.inLan:
            graph.node(self.name, color="blue")
        else:
            graph.node(self.name, color="red")
        if self.left is not None:
            front.append(self.left)
        if self.right is not None:
            front.append(self.right)

        while len(front) != 0:
            v = front.pop(0)
            is_leaf = True

            if v.left is not None:
                front.append(v.left)
                is_leaf = False
            if v.right is not None:
                front.append(v.right)
                is_leaf = False

            if is_leaf:
                name = v.name + "[leaf]"
            else:
                name = v.name

            if v.inLan:
                graph.node(name, color="blue")
            else:
                graph.node(name, color="red")

            if v.parent.left == v:
                graph.edge(v.parent.name, name, color="red", label="l")
            else:
                graph.edge(v.parent.name, name, color="blue", label="r")

        graph.view()


def finding_common_ancestor(node1: TreeNode, node2: TreeNode):
    # bring the nodes to the same depth:
    if node1.depth < node2.depth:
        while node1.depth != node2.depth:
            node2 = node2.parent
    else:
        while node1.depth != node2.depth:
            node1 = node1.parent

    while node1 != node2:
        node1, node2 = node1.parent, node2.parent

    return node1


class DecisionTreeLearner(Learner):
    """
    Implementation of the DFA learner from:
      Michael J. Kearns, Umesh Vazirani - An Introduction to Computational Learning Theory
    """

    def __init__(self, teacher):
        self.teacher = teacher
        self._root = TreeNode(in_lan=teacher.membership_query(tuple()))
        self._leafs = [self._root]
        self.dfa = self._produce_hypothesis()
        self.prev_examples = {}

    def _sift(self, word):
        current_node = self._root
        while True:
            if current_node in self._leafs:
                return current_node

            if self.prev_examples.setdefault(word + current_node.name,
                                             self.teacher.membership_query(word + current_node.name)):
                # if self.teacher.membership_query(word + current_node.name):
                current_node = current_node.right
            else:
                current_node = current_node.left

    def _produce_hypothesis(self):
        transitions = {}
        final_nodes = []
        for leaf in self._leafs:
            if leaf.inLan:
                final_nodes.append(leaf.name)
            tran = {}
            for l in self.teacher.alphabet:
                state = self._sift(leaf.name + tuple([l]))
                tran.update({l: state.name})
            transitions.update({leaf.name: tran})

        return DFA(tuple(""), tuple(final_nodes), transitions)

    def _sift_set(self, words: []):
        """
        Like regular sift but done for a batches of words.
        This is a speeding up for RNN learning.
        """
        words_left = len(words)
        current_nodes = [[self._root, i] for i in range(words_left)]
        final = [None for _ in words]
        while True:
            answers = self.teacher.model.is_words_in_batch([words[x[1]] + x[0].name for x in current_nodes])
            if len(answers.shape) == 0:
                if answers > 0.5:
                    current_nodes[0][0] = current_nodes[0][0].right
                else:
                    current_nodes[0][0] = current_nodes[0][0].left

                if current_nodes[0][0] in self._leafs:
                    final[current_nodes[0][1]] = current_nodes[0][0]
                    del (current_nodes[0])
                    words_left = words_left - 1
            else:
                for i in range(len(answers) - 1, -1, -1):
                    if answers[i] > 0.5:
                        current_nodes[i][0] = current_nodes[i][0].right
                    else:
                        current_nodes[i][0] = current_nodes[i][0].left

                    if current_nodes[i][0] in self._leafs:
                        final[current_nodes[i][1]] = current_nodes[i][0]
                        del (current_nodes[i])
                        words_left = words_left - 1
            if words_left == 0:
                return final

    def _produce_hypothesis_set(self):
        """
            Like regular produce_hypothesis but done for a batches of words.
            This is a speeding up for RNN learning.
        """
        transitions = {}
        final_nodes = []
        leafs_plus_letters = []
        for leaf in self._leafs:
            if leaf.inLan:
                final_nodes.append(leaf.name)
            for letter in self.teacher.alphabet:
                leafs_plus_letters.append(leaf.name + tuple([letter]))
        states = self._sift_set([leafs_plus_letters[0]])
        states = self._sift_set(leafs_plus_letters)
        for leaf in range(len(self._leafs)):
            transition = {}
            for letter in range(len(self.teacher.alphabet)):
                transition.update(
                    {self.teacher.alphabet[letter]: states[leaf * len(self.teacher.alphabet) + letter].name})
            transitions.update({self._leafs[leaf].name: transition})

        return DFA(tuple(""), final_nodes, transitions)

    def new_counterexample(self, word, do_hypothesis_in_batches=False, max_refinements=10):
        val = self.dfa.is_word_in(word)
        numb_of_refinements = 0
        while self.dfa.is_word_in(word) == val:
            if numb_of_refinements >= max_refinements:
                # print("num of ref: {}".format(numb_of_refinements))
                return numb_of_refinements
            numb_of_refinements += 1
            first_time = False
            if len(self._leafs) == 1:
                first_time = True
                new_differencing_string = self._leafs[0].name
                new_state_string = word

            else:
                state_dfa = self.dfa.init_state
                prefix = tuple()
                for letter in word:
                    prefix = prefix + tuple([letter])
                    state_tree = self._sift(prefix)
                    state_dfa = self.dfa.next_state_by_letter(state_dfa, letter)
                    if state_tree.name != state_dfa:
                        for state_tree_2 in self._leafs:
                            if state_tree_2.name == state_dfa:
                                break
                        state_tree = finding_common_ancestor(state_tree, state_tree_2)
                        new_differencing_string = tuple([letter]) + state_tree.name
                        break

                new_state_string = prefix[0:len(prefix) - 1]

            node_to_replace = self._sift(new_state_string)

            if self.teacher.membership_query(node_to_replace.name + new_differencing_string):
                node_to_replace.left = TreeNode(new_state_string, first_time ^ node_to_replace.inLan, node_to_replace)
                node_to_replace.right = TreeNode(node_to_replace.name, node_to_replace.inLan, node_to_replace)
            else:
                node_to_replace.right = TreeNode(new_state_string, first_time ^ node_to_replace.inLan, node_to_replace)
                node_to_replace.left = TreeNode(node_to_replace.name, node_to_replace.inLan, node_to_replace)

            self._leafs.remove(node_to_replace)
            node_to_replace.name = new_differencing_string
            self._leafs.extend([node_to_replace.right, node_to_replace.left])

            if do_hypothesis_in_batches:
                self.dfa = self._produce_hypothesis_set()
            else:
                self.dfa = self._produce_hypothesis()
        # if numb_of_refinements > 1:
        # print("num of ref: {}".format(numb_of_refinements))
        return numb_of_refinements
