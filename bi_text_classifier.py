import string#to remove punctuation
from pathlib import Path
import ntpath as p #to retrieve the name of a file
import sys #to read arguments
from collections import Counter #top 10 words
from tabulate import tabulate #to organize top 10 words in a nice table

def guess(model, doc):
    """ predict a genre of a document based on a trained model.
    this function predicts a positive genre when the sum of polarities + bias
    value is positive and vice versa when it's negative, polarity is provided in
    the passed model (in the parameter).
    Args:
        model: a tuple of size 4, the 1st is a string of the first (positive)
                genre, the 2nd is the second (negative) genre, the third is an
                integer that represents the bias and finally the 4th element is
                a polarity lexicon which is a dictionary that maps a word to its
                polarity.
        doc: a list of strings (words) that the function will guess its genre.
    Returns:
        A string that holds the value of the genre name guessed for a specifc doc
    """
    pos_genre = model[0]
    neg_genre = model[1]
    bias = model[2]
    lex = model[3]
    sum_of_polarities = bias
    for str in doc:
        if str in lex.keys():
            polarity = lex[str]
            sum_of_polarities += polarity

    if sum_of_polarities > 0:
        return pos_genre
    elif sum_of_polarities < 0:
        return neg_genre
    return None


def train(pos_genre, neg_genre, training_data, n):
    """ train a model that can predict pos and neg genres based on training data
    this function updates a model's polarity lex by adding 1 to its words' polarity values
    and the bias when the model predicts correctly, otherwise it reduces 1 from its
    words' polarity values and its bias.
    Args:
        pos_genre: a string that holds the name of the positive genre
        neg_genre: a string that holds the name of the negative genre
        training_data: list of tuples, where each tuple is of size 2 and holds
        the name of the genre with a corresponded document.
        n: number of iterations the function logic should be run.
    Returns:
        A model, which is a 4-element tuple. the 1st element is the name of the
        positive genre name, the 2nd is the negative genre name, the 3rd is an
                integer represents the bias and the 4th is a polarity lexicon
                which is a dictionary that maps a word to its polarity.
    """
    bias = 0
    lex = {}
    for i in range(n):
        for genre, doc in training_data:
            model = (pos_genre, neg_genre, bias, lex)
            guessed_genre = guess(model, doc)
            if guessed_genre is not genre and genre == pos_genre: #if guessed wrong for positive.
                bias += 1
                for word in doc:
                    if word not in lex.keys():
                        lex[word] = 0 #initialize the word with polarity 0.
                    lex[word] +=1
            elif guessed_genre is not genre and genre == neg_genre:
                bias -= 1
                for word in doc:
                    if word not in lex.keys():
                        lex[word] = 0
                    lex[word] -=1
    return model



def test(model,testing_data):
    """ test and evaluate a trained model based on testing data.
    this function uses the guess() function to predict the genre of each document
    in the testing data, then based on the guessed and correct genre, it counts
    the true positives and true negatives.
    Args:
        model: a trained model (tuple) that consists of the positive and negative
        genre names, the bias and a polarity lexicon.
        testing_data: list of tuples, where each tuple is of size 2 and holds
        the name of the genre with a corresponded document to test.
    Returns:
        test_results: a 4-element tuple that consists of the counts of i) true
        positives ii) all positives iii) true negatives iv) all negatives.
    """
    all_genre = list({tuple[0] for tuple in testing_data})
    pos_genre = all_genre[1]
    neg_genre = all_genre[0]
    all_pos = 0
    all_neg = 0
    true_pos = 0
    true_neg = 0
    for genre, doc in testing_data:
        guessed_genre = guess(model, doc)
        if genre == pos_genre:
            all_pos += 1
            if guessed_genre == pos_genre:
                true_pos += 1

        elif genre == neg_genre:
            all_neg += 1
            if guessed_genre == neg_genre:
                true_neg += 1
    test_result = (true_pos, all_pos, true_neg, all_neg)
    return test_result



def preprocess(args):
    """ pre-process raw data from files to fit the train and test algorithms.
    this function reads the content of raw data and i)cleans the data ii)prepare
    training set and interleave them iii) prepare the testing set iv)does the prev
    steps for both repeated and unique data points.
    Args:
        args: the arguments from the command line (beside the program name) that
        should be paths to the raw data files for each genre.
    Returns:
        [to_nlp, to_nlp_unique]: lists of lists that organize the training and
        testing data for both unique and repeated data variations.
    """
    if len(args) < 2:
        print("Run the python script and give it exactly two arguments; pos_genre path and neg_genre path")
        return
    pos_file = Path(args[0])
    neg_file = Path(args[1])

    training_set = 60
    testing_set = 15

    #lists of tuples
    to_train = []
    to_train_unique = []

    to_test = []
    to_test_unique = []
    #lists inside the tuples
    training_data_pos = []
    training_data_pos_unique = []
    training_data_neg = []
    training_data_neg_unique = []

    testing_data_pos = []
    testing_data_pos_unique = []
    testing_data_neg = []
    testing_data_neg_unique = []

    #extracting the names of the genres from the file name, another way to read the genre name from the file content is pos_genre = doc.split("\t", 1)[0]
    pos_genre = p.basename(pos_file).replace('.tsv', '')
    neg_genre = p.basename(neg_file).replace('.tsv', '')

    with open(pos_file) as pos_file, open(neg_file) as neg_file:
        for i in range(training_set):
            #positive genre file
            doc_train_pos = pos_file.readline()
            training_data_pos= clean(doc_train_pos)
            training_data_pos_unique= unique(training_data_pos)

            train_pos = (pos_genre, training_data_pos)
            train_pos_unique = (pos_genre, training_data_pos_unique)

            #negative genre file
            doc_train_neg = neg_file.readline()
            training_data_neg= clean(doc_train_neg)
            training_data_neg_unique= unique(training_data_neg)

            train_neg = (neg_genre, training_data_neg)
            train_neg_unique = (neg_genre, training_data_neg_unique)

            #organize training data
            to_train.extend((train_pos, train_neg))
            to_train_unique.extend((train_pos_unique, train_neg_unique))

        for i in range(testing_set):
            #positive genre file
            doc_test_pos = pos_file.readline()
            testing_data_pos= clean(doc_test_pos)
            testing_data_pos_unique= unique(testing_data_pos)

            test_pos = (pos_genre, testing_data_pos)
            test_pos_unique = (pos_genre, testing_data_pos_unique)

            #negative genre file
            doc_test_neg = neg_file.readline()
            testing_data_neg= clean(doc_test_neg)
            testing_data_neg_unique= unique(testing_data_neg)

            #organize testing data
            test_neg = (neg_genre, testing_data_neg)
            test_neg_unique = (neg_genre, testing_data_neg_unique)

            to_test.extend((test_pos, test_neg))
            to_test_unique.extend((test_pos_unique, test_neg_unique))

        to_nlp = [to_train, to_test]
        to_nlp_unique = [to_train_unique, to_test_unique]

    return [to_nlp, to_nlp_unique]

def clean(data):
    """ clean data by removing proper nouns and punctuation.
    Args:
        data: a big 1 string of raw data
    Returns:
        data: list of strings (words) with no proper nouns nor punctuation.
    """
    data = [word.split("|", 1)[0].lower() for word in data.split(' ') if "NP" not in word] #remove proper nouns
    data = [word.rstrip(string.punctuation) for word in data] #remove punctuation
    data = list(filter(None, data)) #remove empty strings
    return data

def unique(data):
    """ remove duplicates from a list
    this function converts a list to a set then to a list again to get rid of
    duplicate values.
    Args:
        data: list of strings (words)
    Returns:
        data: list of strings (words) with no duplicates
    """
    data = set(data)
    data = list(data)
    return data

def top_10(lex):
    """ Retrieve the top 10 words associates with a positive genre
    Args:
        lex: polarity lexicon (dictionary) that maps a word with its polarity value.
    Returns:
        table: table that hold the top 10 words (with the highest polarity) along
        with their polatity values.
    """
    counter = Counter(lex)
    top10 = counter.most_common(10)  #from python3 Collections documentation: https://docs.python.org/3.7/library/collections.html
    values = [[word, polarity] for word, polarity in top10]
    head = ['Top 10 words (Positive genre)', 'Polarity']
    table = tabulate(values, head, tablefmt="simple") #from https://pypi.org/project/tabulate/
    return table

def least_10(lex):
    """ Retrieve the top 10 words associates with a negaive genre
    Args:
        lex: polarity lexicon (dictionary) that maps a word with its polarity value.
    Returns:
        table: table that hold the top 10 words (with the lowest polarity) along
        with their polatity values.
    """
    counter = Counter(lex)
    top10 = counter.most_common()[:-10-1:-1] #from python3 Collections documentation: https://docs.python.org/3.7/library/collections.html
    values = [[word, polarity] for word, polarity in top10]
    head = ['Top 10 words (Negative genre)', 'Polarity']
    table = tabulate(values, head, tablefmt="simple") #from https://pypi.org/project/tabulate/
    return table

def main():

    to_nlp, to_nlp_unique = preprocess(sys.argv[1:])

    pos_genre = to_nlp[0][0][0]
    neg_genre = to_nlp[0][1][0]

    training_data = to_nlp[0]
    testing_data = to_nlp[1]

    training_data_unique = to_nlp_unique[0]
    testing_data_unique = to_nlp_unique[1]

    print("\n training the model with ", len(training_data), " example documents (with all word occurrences)... \n")
    model1 = train(pos_genre, neg_genre, training_data, 10)
    print(top_10(model1[3]))
    print(least_10(model1[3]))

    print("\n training the model with ", len(training_data_unique), " example documents (with unique word occurrences)... \n")
    model2 = train(pos_genre, neg_genre, training_data_unique, 10)
    print(top_10(model2[3]))
    print(least_10(model2[3]))
    print("\n testing the model with ", len(testing_data), " test example documents (with all word occurrences)... \n")
    test_results = test(model1,testing_data)

    print("\n testing the model with ", len(testing_data), " test example documents (with unique word occurrences)... \n")
    test_results_unique = test(model2,testing_data_unique)

    print("results of the test: (with all word occurrences) ")
    print("True Positives: ", test_results[0], " out of ", test_results[1], " positive examples ")
    print("True Negatives: ", test_results[2], " out of ", test_results[3], " negative examples \n")

    print("results of the test: (with unique word occurrences) ")
    print("True Positives: ", test_results_unique[0], " out of ", test_results_unique[1], " positive examples")
    print("True Negatives: ", test_results_unique[2], " out of ", test_results_unique[3], " negative examples")


if __name__=='__main__':
    main();
