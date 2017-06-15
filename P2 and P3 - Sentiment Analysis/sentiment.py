import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    features = []

    word_in_pos = {}
    word_in_neg = {}

    for text in train_pos:
        for word in set(text):
            if word not in stopwords:
                if word in word_in_pos:
                    word_in_pos[word] += 1
                else:
                    word_in_pos[word] = 1
                    
    for text in train_neg:
        for word in set(text):
            if word not in stopwords:
                if word in word_in_neg:
                    word_in_neg[word] += 1
                else:
                    word_in_neg[word] = 1
                    
    pos_threshold = 0.01 * len(train_pos)
    neg_threshold = 0.01 * len(train_neg)

    for word, count in word_in_pos.iteritems():
        if count >= pos_threshold and 2 * word_in_neg[word] <= count:
            features.append(word)

    for word, count in word_in_neg.iteritems():
        if count >= neg_threshold and 2 * word_in_pos[word] <= count and word not in features:
            features.append(word)

    train_pos_vec = binary_text(train_pos, features)
    train_neg_vec = binary_text(train_neg, features)
    test_pos_vec = binary_text(test_pos, features)
    test_neg_vec = binary_text(test_neg, features) 
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def binary_text(data, features):
    binary_vec_total = []
    for text in data:
        words = set(text)
        binary_vec = []
        for word in features:
            if word in words:
                binary_vec.append(1)
            else:
                binary_vec.append(0)
        binary_vec_total.append(binary_vec)
    return binary_vec_total

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """

    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []

    i = 0
    for text in train_pos:
        words = set(text)
        ls = LabeledSentence(words, ['TRAIN_POS_%s' % i])
        labeled_train_pos.append(ls)
        i += 1
        
    i = 0
    for text in train_neg:
        words = set(text)
        ls = LabeledSentence(words, ['TRAIN_NEG_%s' % i])
        labeled_train_neg.append(ls)
        i += 1
        
    i = 0
    for text in test_pos:
        words = set(text)
        ls = LabeledSentence(words, ['TEST_POS_%s' % i])
        labeled_test_pos.append(ls)
        i += 1
        
    i = 0
    for text in test_neg:
        words = set(text)
        ls = LabeledSentence(words, ['TEST_NEG_%s' % i])
        labeled_test_neg.append(ls)
        i += 1


    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    
    train_pos_vec = []
    for i in range(len(train_pos)):
        train_pos_vec.append(model.docvecs['TRAIN_POS_%s' % i])

    train_neg_vec = []
    for i in range(len(train_neg)):
        train_neg_vec.append(model.docvecs['TRAIN_NEG_%s' % i])

    test_pos_vec = []
    for i in range(len(test_pos)):
        test_pos_vec.append(model.docvecs['TEST_POS_%s' % i])

    test_neg_vec = []
    for i in range(len(test_neg)):
        test_neg_vec.append(model.docvecs['TEST_NEG_%s' % i])


    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec
        
def create_labeled_sentence(data):
    labeled = []
    i = 0
    for text in data:
        words = set(text)
        ls = LabeledSentence(words, ['TRAIN_POS_%s' % i])
        labeled.append(ls)
        i += 1
    return labeled

def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    train_data = train_pos_vec + train_neg_vec

    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(train_data, Y)
    
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(train_data, Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    train_data = train_pos_vec + train_neg_vec

    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(train_data, Y)
    
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(train_data, Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    
    pos_predictions = model.predict(test_pos_vec)
    neg_predictions = model.predict(test_neg_vec)
    
    total_pos = len(test_pos_vec)
    total_neg = len(test_neg_vec)
    tp = sum([1 if prediction == "pos" else 0 for prediction in pos_predictions])
    tn = sum([1 if prediction == "neg" else 0 for prediction in neg_predictions])
    fp = abs(tn - total_neg)
    fn = abs(tp - total_pos)
    accuracy = float(tp + tn) / (total_pos + total_neg)
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
