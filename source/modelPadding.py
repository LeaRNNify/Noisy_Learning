import os
import time
from copy import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class WordsDataset(Dataset):
    def __init__(self, words, labels):
        self.labels = labels
        self.words = words

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.words[idx]), self.labels[idx]


def teach(model, batch_size, train_loader, val_loader, device, lr=0.005, criterion=nn.BCELoss(),
          epochs=10, print_every=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    counter = 0
    clip = 5
    valid_loss_min = np.Inf
    last_loss = 1
    print("Num of training examples: {}".format(len(train_loader) * batch_size))
    print("Begin training: ")
    start_time = time.time()
    for i in range(epochs):
        h = model.init_hidden(batch_size)

        val_h = model.init_hidden(batch_size)
        val_losses = []
        model.eval()
        num_correct = 0
        for inp, lab, inp_len in val_loader:
            # val_h = tuple([each.data for each in val_h])
            inp, lab = inp.to(device), lab.to(device)
            out, _ = model(inp, inp_len, val_h)
            val_loss = criterion(out.squeeze(), lab.float())
            val_losses.append(val_loss.item())

            pred = torch.round(out.squeeze())  # rounds the output to 0/1
            correct_tensor = pred.eq(lab.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)

        test_acc = num_correct / len(val_loader.dataset) * 100
        print("Summary for after Epoch {}/{}:".format(i, epochs))
        print("Val Loss: {:.6f}".format(np.mean(val_losses)),
              "Test accuracy: {:.3f}%".format(test_acc),
              "Loss delta: {:.6f}".format(last_loss - np.mean(val_losses)),
              "Time: {:.0f}".format(time.time() - start_time))
        start_time = time.time()
        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("Starting Epoch {}/{}".format(i + 1, epochs))
        if 0.0009 < last_loss - np.mean(val_losses) < 0.005:
            lr = last_loss - np.mean(val_losses)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print("changed to learning rate: {}".format(lr))
        # if (0.01 < last_loss - np.mean(val_losses)) & (lr != 0.005):
        #     lr = 0.005
        #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #     print("changed to learning rate: {}".format(lr))

        last_loss = np.mean(val_losses)

        model.train()
        for inputs, labels, inp_len in train_loader:
            counter += 1
            # h = tuple([e.data for e in h])

            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, _ = model(inputs, inp_len, h)
            loss = criterion(output.squeeze(), labels.float())
            # batch_ce_loss = 0.0
            # for i in range(output.size(0)):
            #     j = output[i][inp_len[i] - 1]
            #     ce_loss = cross_entropy(j, labels[i])
            #     batch_ce_loss += ce_loss

            loss.backward()
            # batch_ce_loss
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inp, lab, inp_len in val_loader:
                    # val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, _ = model(inp, inp_len, val_h)
                    # batch_ce_loss = 0.0
                    # for i in range(output.size(0)):
                    #     ce_loss = cross_entropy(output[i][inp_len[i] - 1].squeeze(), labels[i])
                    #     batch_ce_loss += ce_loss
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), './state_dict.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                    np.mean(
                                                                                                        val_losses)))
                    valid_loss_min = np.mean(val_losses)
                    # if valid_loss_min < 0.00005:
                    #     return test_acc
    return test_acc


def test_rnn(model, test_loader, batch_size, device, criterion=nn.BCELoss()):
    # Loading the best model
    model.load_state_dict(torch.load('./state_dict.pt'))

    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)

    model.eval()
    for inputs, labels, len_inp in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, _ = model(inputs, len_inp, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc: float = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))
    return test_acc * 100


def from_array_to_word(int2char, array):
    word = []
    for i in array:
        word.append(int2char[i])
    return word


def pad_collate(batch):
    (xx, yy) = (zip(*batch))
    # a = xx.unique()

    x_lens = [len(x) for x in xx]
    # y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    # yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, torch.tensor(yy), x_lens  # , y_lens


def make_training_set(alphabet, target, num_of_exm_per_length=2000, max_length=50,
                      batch_size=50):
    int2char = ({i + 1: alphabet[i] for i in range(len(alphabet))})
    int2char.update({0: ""})
    char2int = {alphabet[i]: i + 1 for i in range(len(alphabet))}
    char2int.update({"": 0})

    train = create_words_set(alphabet, batch_size, int2char, max_length, num_of_exm_per_length, target)

    validation = create_words_set(alphabet, batch_size, int2char, max_length, int(num_of_exm_per_length / 5), target)

    val_length = int(len(validation) // batch_size * 0.4) * batch_size
    val, test = torch.utils.data.random_split(validation, [val_length, len(validation) - val_length])

    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)
    val_loader = DataLoader(val, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)
    test_loader = DataLoader(test, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)

    return train_loader, val_loader, test_loader


def make_training_set_sampler(alphabet, target, sampler, num_of_examples, batch_size=50):
    int2char = ({i + 1: alphabet[i] for i in range(len(alphabet))})
    int2char.update({0: ""})
    char2int = {alphabet[i]: i + 1 for i in range(len(alphabet))}
    char2int.update({"": 0})

    train = create_words_set_sampler(alphabet, batch_size, int2char, char2int, target, sampler, num_of_examples)

    validation = create_words_set_sampler(alphabet, batch_size, int2char, char2int, target, sampler,
                                          num_of_examples / 10)

    val_length = int(len(validation) // batch_size * 0.4) * batch_size
    val, test = torch.utils.data.random_split(validation, [val_length, len(validation) - val_length])

    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)
    val_loader = DataLoader(val, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)
    test_loader = DataLoader(test, shuffle=True, batch_size=batch_size, collate_fn=pad_collate)

    return train_loader, val_loader, test_loader


def create_words_set_sampler(alphabet, batch_size, int2char, char2int, target, sampler, num_of_examples):
    words_list = [np.array([char2int[l] for l in sampler(alphabet)]) for _ in range(int(num_of_examples))]

    round_num_batches = int(len(words_list) - len(words_list) % batch_size)
    words_list = words_list[:round_num_batches - 1]
    label_list = [target(from_array_to_word(int2char, w)) for w in words_list]

    print("Positive examples: {:.3}".format(sum([int(lab) for lab in label_list]) / len(words_list)))

    if sum([int(lab) for lab in label_list]) < 0.05 * len(words_list):
        print("not enough positive examples")
        print("before: {:.3}".format(sum([int(lab) for lab in label_list]) / len(words_list)))
        add_examples_with_specific_label_sampler(alphabet, label_list, int(num_of_examples), target,
                                                 words_list, True, sampler, char2int)
        print("after: {:.3}".format(sum([int(lab) for lab in label_list]) / len(words_list)))

    elif sum([int(not lab) for lab in label_list]) < 0.05 * len(words_list):
        print("not enough positive examples")
        print("before: {:.3}".format(sum([int(not lab) for lab in label_list]) / len(words_list)))
        add_examples_with_specific_label_sampler(alphabet, label_list, int(num_of_examples), target,
                                                 words_list, False, sampler, char2int)
        print("after: {:.3}".format(sum([int(not lab) for lab in label_list]) / len(words_list)))

    words_list = [w if len(w) != 0 else np.array([0]) for w in words_list]

    # words_list.insert(0, np.array([0]))
    # label_list.insert(0, target(""))

    return WordsDataset(words_list, label_list)


def create_words_set(alphabet, batch_size, int2char, max_length, num_of_exm_per_length, target):
    words_list = []
    lengths = list(range(1, max_length))  # + list(range(20, max_length, 5))
    for length in lengths:
        new_list = np.unique(np.random.randint(1, len(alphabet) + 1, size=(num_of_exm_per_length, length)), axis=0)
        words_list.extend(new_list)
    round_num_batches = int(len(words_list) - len(words_list) % batch_size)
    words_list = words_list[:round_num_batches - 1]
    label_list = [target(from_array_to_word(int2char, w)) for w in words_list]

    print("Positive examples: {:.3}".format(sum([int(lab) for lab in label_list]) / len(words_list)))

    if sum([int(lab) for lab in label_list]) < 0.05 * len(words_list):
        print("not enough positive examples")
        print("before: {:.3}".format(sum([int(lab) for lab in label_list]) / len(words_list)))
        add_examples_with_specific_label(alphabet, int2char, label_list, max_length, num_of_exm_per_length, target,
                                         words_list, True)
        print("after: {:.3}".format(sum([int(lab) for lab in label_list]) / len(words_list)))

    elif sum([int(not lab) for lab in label_list]) < 0.05 * len(words_list):
        print("not enough positive examples")
        print("before: {:.3}".format(sum([int(not lab) for lab in label_list]) / len(words_list)))
        add_examples_with_specific_label(alphabet, int2char, label_list, max_length, num_of_exm_per_length, target,
                                         words_list, False)
        print("after: {:.3}".format(sum([int(not lab) for lab in label_list]) / len(words_list)))

    words_list.insert(0, np.array([0]))
    label_list.insert(0, target(""))

    return WordsDataset(words_list, label_list)


def add_examples_with_specific_label_sampler(alphabet, label_list, num_of_examples, target,
                                             words_list, label, sampler, char2int):
    new_examples = []
    max_tries, current_try = 10, 0

    while len(new_examples) < (0.05 * len(words_list)) and current_try < max_tries:
        words = set()
        for _ in range(num_of_examples):
            w = sampler(alphabet)
            if target(w) == label:
                words.add(w)
        new_examples.extend([np.array([char2int[l] for l in w]) for w in words])
        current_try += 1

    new_examples_value = [copy(label) for _ in new_examples]
    words_list.extend(new_examples)
    label_list.extend(new_examples_value)


def add_examples_with_specific_label(alphabet, int2char, label_list, max_length, num_of_exm_per_length, target,
                                     words_list, label):
    new_examples = []
    len_of_saturated_words = int(np.log(num_of_exm_per_length) / np.log(len(alphabet)))
    max_tries, current_try = 20, 0

    while len(new_examples) < (0.03 * len(words_list)) and current_try < max_tries:
        words = []
        for length in range(len_of_saturated_words, max_length + current_try):
            words.extend(
                np.unique(np.random.randint(1, len(alphabet) + 1, size=(int(num_of_exm_per_length * 10), length)),
                          axis=0))
        new_examples.extend([w for w in words if target(from_array_to_word(int2char, w)) == label])
        # the following is a not optimized shitty procedure that needs to
        unique = list()
        for exm in new_examples:
            add_exmp = True
            for exm_u in unique:
                if len(exm) != len(exm_u):
                    continue
                if all(np.array(exm_u) == np.array(exm)):
                    add_exmp = False
                    break
            if add_exmp:
                unique.append(exm)

        new_examples = unique
        current_try += 1

    new_examples_value = [copy(label) for _ in new_examples]
    words_list.extend(new_examples)
    label_list.extend(new_examples_value)


class LSTM(nn.Module):
    def __init__(self, alphabet_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5,
                 device=torch.device("cpu")):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(alphabet_size, embedding_dim).to(device=device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True).to(device=device)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_size).to(device=device)
        self.sigmoid = nn.Sigmoid().to(device=device)
        self.device = device

    def forward(self, x, x_lens, hidden):
        batch_size = x.size(0)
        x = x.long()
        x_embed = self.embedding(x)

        x_packed = pack_padded_sequence(x_embed, x_lens, batch_first=True, enforce_sorted=False)
        output_padded, hidden = self.lstm(x_packed, hidden)

        out_ltsm, output_lengths = pad_packed_sequence(output_padded, batch_first=True)
        out_ltsm = out_ltsm.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(out_ltsm)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        # outb = torch.tensor([out[i][output_lengths[i] - 1] for i in range(20)])
        outc = out[:, -1]
        output_lengths = (output_lengths - 1).to(self.device)
        outb = out.gather(1, output_lengths.view(-1, 1)).squeeze()

        return outb, hidden

    def init_hidden(self, batch_size):
        # weight = next(self.parameters()).data
        # hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
        #           weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
        return hidden


class RNNLanguageClasifier:
    def __init__(self):
        self._rnn = None
        self._initial_state = None
        self._current_state = None
        self._char_to_int = None
        self.alphabet = []
        self.word_traning_length = 40
        self.num_of_train = 0
        self.num_of_test = 0
        self.test_acc = 0
        self.val_acc = 0
        self.extra_time = 0
        self.num_of_membership_queries = 0

    def train_a_lstm(self, alphahbet, target, sampler, embedding_dim=10, hidden_dim=10, num_layers=2, batch_size=20,
                     num_of_examples=5000, word_traning_length=40, epoch=20):
        self.word_traning_length = word_traning_length
        self._char_to_int = {alphahbet[i]: i + 1 for i in range(len(alphahbet))}
        self._char_to_int.update({"": 0})
        self.alphabet = alphahbet

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        self._rnn = LSTM(len(alphahbet) + 1, 1, embedding_dim, hidden_dim, num_layers, drop_prob=0.5,
                         device=device)
        # make_training_set(alphahbet, target)
        train_loader, val_loader, test_loader = \
            make_training_set_sampler(alphahbet, target, sampler, num_of_examples, batch_size)

        self.num_of_train, self.num_of_test = len(train_loader) * batch_size, len(test_loader) * batch_size

        self.val_acc = teach(self._rnn, batch_size, train_loader, val_loader, device, epochs=epoch, print_every=100)

        self._initial_state = self._rnn.init_hidden(1)
        self._current_state = self._initial_state

        self.test_acc = test_rnn(self._rnn, test_loader, batch_size, device)
        self.states = {
            str(self.from_state_to_list(self._rnn.init_hidden(1))): ""}
        return

    def is_word_in(self, word):
        self.num_of_membership_queries += 1
        h = self._rnn.init_hidden(1)
        if len(word) == 0:
            length = torch.tensor([1]).to(self._rnn.device)
            array = torch.tensor([[0]]).to(self._rnn.device)
        else:
            length = torch.tensor([len(word)]).to(self._rnn.device)
            array = torch.tensor([[self._char_to_int[l] for l in word]]).to(self._rnn.device)
        output, h = self._rnn(array, length, h)
        return bool(output > 0.5)

    def is_words_in_batch(self, words):
        self.num_of_membership_queries += len(words)
        words_torch = [torch.tensor([self._char_to_int[l] for l in word]) if len(word) != 0 else torch.tensor([0])
                       for word in words]
        # for word in words:
        #     lengths.append(len(word))
        #     words_in_num.append([self._char_to_int[l] for l in word])
        h = self._rnn.init_hidden(len(words))
        # words, lengths, _ = pad_collate((words, [0]*len(words)))

        x_lens = [len(word) if len(word) != 0 else 1 for word in words]
        # y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(words_torch, batch_first=True, padding_value=0).to(self._rnn.device)

        output, _ = self._rnn(xx_pad, x_lens, h)
        return output

    def reset_current_to_init(self):
        self._current_state = self._rnn.init_hidden(1)

    def save_lstm(self, dir_name, force_overwrite=False):
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        elif os.path.exists(dir_name + "/meta") & (not force_overwrite):
            if input(
                    "there already exists a model in {}. Enter y if you want to overwrite it.".format(dir_name)) != "y":
                return
        with open(dir_name + "/meta", "w+") as file:
            file.write("Metadata:\n")
            file.write("alphabet = ")
            file.write(str(self.alphabet[0]))
            for l in range(len(self.alphabet) - 1):
                file.write("," + str(self.alphabet[l + 1]))
            file.write("\n")
            file.write("embedding_dim = " + str(self._rnn.embedding_dim) + "\n")
            file.write("hidden_dim = " + str(self._rnn.hidden_dim) + "\n")
            file.write("n_layers = " + str(self._rnn.n_layers) + "\n")
            file.write("torch_save = state_dict.pt")
        torch.save(self._rnn.state_dict(), dir_name + "/state_dict.pt")

    def load_lstm(self, dir_name):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        with open(dir_name + "/meta", "r") as file:
            for line in file.readlines():
                splitline = line.split(" = ")
                if splitline[0] == "alphabet":
                    self.alphabet = splitline[1].rstrip('\n').split(",")
                elif splitline[0] == "embedding_dim":
                    embedding_dim = int(splitline[1])
                elif splitline[0] == "hidden_dim":
                    hidden_dim = int(splitline[1])
                elif splitline[0] == "n_layers":
                    n_layers = int(splitline[1])
                elif splitline[0] == "torch_save":
                    torch_save = splitline[1].rstrip('\n')

        self._rnn = LSTM(len(self.alphabet) + 1, 1, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, device=device)
        self._rnn.load_state_dict(torch.load(dir_name + "/" + torch_save, map_location={'cuda:0': 'cpu'}))
        self._rnn.eval()
        torch.no_grad()

        self._initial_state = self._rnn.init_hidden(1)
        self._current_state = self._initial_state
        self._char_to_int = {self.alphabet[i]: i + 1 for i in range(len(self.alphabet))}
        self._char_to_int.update({"": 0})
        self.states = {
            str(self.from_state_to_list(self._rnn.init_hidden(1))): ""}  # maybe move to load? or some other place?
        return self

    ######################################################
    #                 Code For Lstar                     #
    ######################################################

    def classify_word(self, word):
        return bool(self.is_word_in(word))

    def get_first_RState(self):
        start_time = time.time()
        list_state = self.from_state_to_list(self._rnn.init_hidden(1))
        self.extra_time = self.extra_time + time.time() - start_time
        return list_state, bool(self.is_word_in(""))

    def get_next_RState(self, state, char):
        start_time = time.time()
        word = self.states[str(state)] + char
        h = self._rnn.init_hidden(1)
        if len(word) == 0:
            length = torch.tensor([1]).to(self._rnn.device)
            array = torch.tensor([[0]]).to(self._rnn.device)
        else:
            length = torch.tensor([len(word)]).to(self._rnn.device)
            array = torch.tensor([[self._char_to_int[l] for l in word]]).to(self._rnn.device)
        output, h = self._rnn(array, length, h)

        state_list = self.from_state_to_list(h)
        self.states.update({str(state_list): word})
        self.extra_time = self.extra_time + time.time() - start_time
        return state_list, bool(output > 0.5)

    def from_state_to_list(self, state):
        list_state = []
        for i in state[0][0, 0]:
            list_state.append(float(i))
        for i in state[1][0, 0]:
            list_state.append(float(i))

        return list_state

    def from_list_to_state(self, list_state):
        hiden = torch.tensor([[list_state[self._rnn.hidden_dim:]]])
        cell = torch.tensor([[list_state[:self._rnn.hidden_dim]]])
        return (hiden.to(self._rnn.device), cell.to(self._rnn.device))
