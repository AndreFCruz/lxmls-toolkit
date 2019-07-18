from __future__ import division
import sys
import numpy as np
import lxmls.sequences.discriminative_sequence_classifier as dsc
import pdb


class StructuredPerceptron(dsc.DiscriminativeSequenceClassifier):
    """ Implements Structured Perceptron"""

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 num_epochs=10, learning_rate=1.0, averaged=True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []

    def train_supervised(self, dataset):
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        num_examples = dataset.size()
        for epoch in range(self.num_epochs):
            num_labels_total = 0
            num_mistakes_total = 0
            for i in range(num_examples):
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
            self.params_per_epoch.append(self.parameters.copy())
            acc = 1.0 - num_mistakes_total / num_labels_total
            print("Epoch: %i Accuracy: %f" % (epoch, acc))
        self.trained = True

        if self.averaged:
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w /= len(self.params_per_epoch)
            self.parameters = new_w

    def perceptron_update(self, sequence):
        """
        General perceptron update rule:
            w_t+1 <- w_t + learning_rate * ( f(seq, y_true) - f(seq, y_pred) )

        Instead of updating the whole feature vector, update each feature individually,
        as needed (when prediction is wrong).
            -> this is done for better efficiency, as updates are very sparse.

        1. At each time-step, check appropriate features (f_init for t=1, f_final for t=N,
            f_emission for t=1...N, f_transition for bi-grams from t=2...N);
        2. If prediction at a given time-step is wrong, subtract to weights/parameters of
            predicted features and sum to weights/parameters of true features;
        """

        # ----------
        # Solution to Exercise 3

        num_mistakes = 0
        length = len(sequence.x)
        pred_seq, _ = self.viterbi_decode(sequence)

        y_true = sequence.y
        y_pred = pred_seq.y

        # Update weights if y_true != y_pred
        y_t_true = y_true[0]
        y_t_pred = y_pred[0]

        # Update initial_features
        if y_t_true != y_t_pred:
            num_mistakes += 1
            true_f_init = self.feature_mapper.get_initial_features(sequence, y_t_true)
            pred_f_init = self.feature_mapper.get_initial_features(sequence, y_t_pred)

            self.parameters[true_f_init] += self.learning_rate
            self.parameters[pred_f_init] -= self.learning_rate

        for i in range(length):
            y_t_true = y_true[i]
            y_t_pred = y_pred[i]

            # Update emission_features
            if y_t_true != y_t_pred:
                num_mistakes += 1
                true_f_emiss = self.feature_mapper.get_emission_features(sequence, i, y_t_true)
                pred_f_emiss = self.feature_mapper.get_emission_features(sequence, i, y_t_pred)
                self.parameters[true_f_emiss] += self.learning_rate
                self.parameters[pred_f_emiss] -= self.learning_rate

            # Update transition features
            if i > 0:   # for each bi-gram, update bigram features if prediction is wrong
                prev_y_t_true = y_true[i-1]
                prev_y_t_pred = y_pred[i-1]

                true_f_trans = self.feature_mapper.get_transition_features(sequence, i, y_t_true, prev_y_t_true)
                pred_f_trans = self.feature_mapper.get_transition_features(sequence, i, y_t_pred, prev_y_t_pred)

                self.parameters[true_f_trans] += self.learning_rate
                self.parameters[pred_f_trans] -= self.learning_rate

        # Update final_features
        y_final_true = y_true[-1]
        y_final_pred = y_pred[-1]
        if y_final_true != y_final_pred:
            num_mistakes += 1
            true_f_final = self.feature_mapper.get_final_features(sequence, y_final_true)
            pred_f_final = self.feature_mapper.get_final_features(sequence, y_final_pred)

            self.parameters[true_f_final] += self.learning_rate
            self.parameters[pred_f_final] -= self.learning_rate

        # return num_mistakes
        return length, num_mistakes

        # End of Solution to Exercise 3
        # ----------

    def save_model(self, dir):
        fn = open(dir + "parameters.txt", 'w')
        for p_id, p in enumerate(self.parameters):
            fn.write("%i\t%f\n" % (p_id, p))
        fn.close()

    def load_model(self, dir):
        fn = open(dir + "parameters.txt", 'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
