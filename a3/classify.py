"""
Assignment 3. Implement a Multinomial Naive Bayes classifier for spam filtering.

You'll only have to implement 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

"""

from collections import defaultdict
import glob
import math
import os



class Document(object):
    """ A Document. Do not modify.
    The instance variables are:

    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename=None, label=None, tokens=None):
        """ Initialize a document either from a file, in which case the label
        comes from the file name, or from specified label and tokens, but not
        both.
        """
        if label: # specify from label/tokens, for testing.
            self.label = label
            self.tokens = tokens
        else: # specify from file.
            self.filename = filename
            self.label = 'spam' if 'spmsg' in filename else 'ham'
            self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):

    def get_word_probability(self, label, term):
        """
        Return Pr(term|label). This is only valid after .train has been called.

        Params:
          label: class label.
          term: the term
        Returns:
          A float representing the probability of this term for the specified class.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_word_probability('spam', 'a')
        0.25
        >>> nb.get_word_probability('spam', 'b')
        0.375
        """
        wp = self.cp[term][label]
        return wp
        pass

    def get_top_words(self, label, n):
        """ Return the top n words for the specified class, using the odds ratio.
        The score for term t in class c is: p(t|c) / p(t|c'), where c'!=c.

        Params:
          labels...Class label.
          n........Number of values to return.
        Returns:
          A list of (float, string) tuples, where each float is the odds ratio
          defined above, and the string is the corresponding term.  This list
          should be sorted in descending order of odds ratio.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_top_words('spam', 2)
        [(2.25, 'b'), (1.5, 'a')]
        """
        classes = []
        for class_ in self.classes:
            if class_ != label:
                classes.append(class_)
        dict = defaultdict(lambda: 0)
        for term in self.vocabulary:
            sum = 0
            for class__ in classes:
                sum = sum + self.cp[term][class__]
            dict[term] = self.cp[term][label] / sum
        result = []
        for k in dict:
            result.append((dict[k], k))
        result = sorted(result, key = lambda x: dict[x[1]], reverse = True)
        return result[:n]
        pass

    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your
        book. Store these as instance variables, to be used by the classify
        method subsequently.
        Params:
          documents...A list of training Documents.
        Returns:
          Nothing.
        """
        self.vocabulary, self.classes = [], []
        self.prior = defaultdict(lambda: 0)
        self.cp = defaultdict(lambda: defaultdict(lambda: 0))
        for d in documents:
            for token in d.tokens:
                if not token in self.vocabulary:
                    self.vocabulary.append(token)
            if not d.label in self.classes:
                self.classes.append(d.label)
        for clas in self.classes:
            tc = defaultdict(lambda: 0)
            docC = []
            for d in documents:
                if d.label == clas:
                    docC.append(d)
            self.prior[clas] = len(docC) / len(documents)
            textC = []
            for d in docC:
                textC += d.tokens
            for term in self.vocabulary:
                tc[term] = textC.count(term)
            l = list(tc.values())
            sum_ = sum(l) + len(l)
            for term in self.vocabulary:
                self.cp[term][clas] = (tc[term] + 1) / sum_
        pass

    def classify(self, documents):
        """ Return a list of strings, either 'spam' or 'ham', for each document.
        Params:
          documents....A list of Document objects to be classified.
        Returns:
          A list of label strings corresponding to the predictions for each document.
        """
        result = []
        for d in documents:
            score = defaultdict(lambda: 0)
            words = []
            for term in d.tokens:
                if not term in words:
                    words.append(term)
            for class_ in self.classes:
                score[class_] = self.prior[class_]
                for term in words:
                    score[class_] *= self.cp[term][class_]
            score = [math.log10(x) for x in score]
            l = list(score.keys())
            maxi = max(l, key = lambda x: score[x])
            result.append(maxi)
        return result
        pass

def evaluate(predictions, documents):
    """ Evaluate the accuracy of a set of predictions.
    Return a tuple of three values (X, Y, Z) where
    X = percent of documents classified correctly
    Y = number of ham documents incorrectly classified as spam
    X = number of spam documents incorrectly classified as ham

    Params:
      predictions....list of document labels predicted by a classifier.
      documents......list of Document objects, with known labels.
    Returns:
      Tuple of three floats, defined above.
    """
    num_corr = 0
    num_ham_incorr = 0
    num_spam_incorr = 0
    for d in documents:
        if predictions[0] == d.label:
            num_corr += 1
        elif d.label == 'ham':
            num_ham_incorr += 1
        else:
            num_spam_incorr += 1
        predictions.pop(0)
    X = num_corr/len(predictions)
    Y = num_ham_incorr
    Z = num_spam_incorr
    return (X, Y, Z)
    pass

def main():
    """ Do not modify. """
    if not os.path.exists('train'):  # download data
       from urllib.request import urlretrieve
       import tarfile
       urlretrieve('http://cs.iit.edu/~culotta/cs429/lingspam.tgz', 'lingspam.tgz')
       tar = tarfile.open('lingspam.tgz')
       tar.extractall()
       tar.close()
    train_docs = [Document(filename=f) for f in glob.glob("train/*.txt")]
    print('read', len(train_docs), 'training documents.')
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(filename=f) for f in glob.glob("test/*.txt")]
    print('read', len(test_docs), 'testing documents.')
    predictions = nb.classify(test_docs)
    results = evaluate(predictions, test_docs)
    print('accuracy=%.3f, %d false spam, %d missed spam' % (results[0], results[1], results[2]))
    print('top ham terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('ham', 10)))
    print('top spam terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('spam', 10)))

if __name__ == '__main__':
    main()
