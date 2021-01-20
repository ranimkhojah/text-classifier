import string #to remove punctuation
from pathlib import Path
import ntpath as p #to retrieve the name of a file
import sys #to read arguments and exit the system
from collections import Counter #top 10 words
from tabulate import tabulate #to organize top 10 words in a nice table
from numpy import * #to use reshape and print the matrix in a structured form

def guess(submodels, doc):
    """ predict a genre of a document based on list of submodels.
    this function predicts a certain genre when the sum of polarities + bias
    value in a given genre is the highest among all genres.
    Args:
        submodels: a list of submodels where each model is a tuple of size 4,
                the 1st is the name of the genre, the second is an integer that
                 represents the bias and finally the 3th element is a polarity
                 lexicon which is a dictionary that maps a word to its polarity.
        doc: a list of strings (words) that the function will guess its genre.
    Returns:
        A string that holds the value of the genre name guessed for a specifc doc.
    """
    highest_polarity = -9999999999 #Dirty way of doing Integer.MIN_VALUE
    guessed_genre = None
    for model in submodels:
        genre = model[0]
        bias = model[1]
        lex = model[2]
        sum_of_polarities = bias

        for str in doc:
            if str in lex.keys():
                polarity = lex[str]
                sum_of_polarities += polarity

        if sum_of_polarities > highest_polarity:
            highest_polarity = sum_of_polarities
            guessed_genre = genre
    return guessed_genre


def train(genres, training_data, n):
    """ train a model that can predict a genre based on training data
    this function updates n submodels' polarity lex (where n is the number of
    genres) by adding 1 to its words' polarity values and the bias when the submodel
    predicts correctly, and vice versa to the submodel that correspods to the
    mistakenly predicted genre (reduce 1 from bias and lex values).
    Args:
        genres: list of all the genre names the submodels list should predict.
        training_data: list of tuples, where each tuple is of size 2 and holds
        the name of the genre with a corresponded document.
        n: number of iterations the function logic should be run (num of epochs).
    Returns:
        A multi-genre model, that is a list of submodels, where each of them is
        a 4-element tuple. the 1st element is the genre name, the 2nd is the bias
        and the 3rd is a polarity lex (dictionary) that maps a word to its polarity.
    """
    submodels = []
    for genre_name in genres:
        model = (genre_name, 0, {}) #initialize submodels with empty models i.e. 0 bias and empty lex
        submodels.append(model)
    for i in range(n):
        # print("Epoch No.", i)
        for genre, doc in training_data:
            guessed_genre = guess(submodels, doc)
            # print("in training: ", guessed_genre, genre)
            if guessed_genre != genre: #if guessed wrong
                for j, submodel in enumerate(submodels):
                    genre_name = submodels[j][0]
                    bias = submodels[j][1]
                    lex = submodels[j][2]
                    if genre_name == genre: #for the correct genre
                        bias += 1 #bias
                        for word in doc:
                            if word not in lex.keys():
                                lex[word] = 0 #initialize the word with polarity 0.
                            lex[word] +=1
                        # print(lex)
                    elif genre_name != genre and guessed_genre == genre_name:
                        bias -= 1
                        for word in doc:
                            if word not in lex.keys():
                                lex[word] = 0 #initialize the word with polarity 0.
                            lex[word] -=1
                    submodels[j] = (genre_name, bias, lex) #update the submodel
    return submodels



def test(submodels,testing_data):
    """ test and evaluate a trained multi-genre model based on testing data.
    this function uses the guess() function to predict the genre of each document
    in the testing data, then based on the guessed and correct genre, it counts
    what genres where correctly and mistakenly predicted using a (n x n) confusion
     matrix, where n is the number of genres/submodels.
    Args:
        submodels: a trained multi-genre model (a list of submodels) where each
        model is a tuple that consists of the genre name, the bias and a polarity lexicon.
        testing_data: list of tuples, where each tuple is of size 2 and holds
        the name of the genre with a corresponded document to test.
    Returns:
        confusion_matrix: a (n x n) matrix, the y-axis is the correct genres and
        the x-axis is the guessed genres, so that the value of confusion_matrix[i][j]
        is the number of times the multi-genre model predicted the genre at
        position [i] as the genre at position [j].
    """
    genres = [submodel[0] for submodel in submodels]
    confusion_matrix = [[0] * len(genres)] * len(genres) #initializing the confusion matrix
    confusion_matrix = reshape(confusion_matrix,(len(genres),len(genres)))
    for genre, doc in testing_data:
        guessed_genre = guess(submodels, doc)
        for i, correct in enumerate(submodels):
            for j, guessed in enumerate(submodels):
                if i == j and guessed_genre == genre and genres[i] == genre and  genres[j] == guessed_genre: #if it predicted correctly, add 1 to the guessed genre
                    confusion_matrix[i][j] += 1
                elif i != j and guessed_genre != genre and genres[i] == genre and genres[j] == guessed_genre: #otherwise, if it doesn't, add 1 to the correct genre
                    confusion_matrix[i][j] += 1

    return confusion_matrix



def preprocess(args):
    """ pre-process raw data from files to fit the train and test algorithms.
    this function reads the content of raw data and i)cleans the data ii)prepares
    training set and interleave them iii) prepare the testing set iv)does the prev
    steps for both repeated and unique data points.
    Args:
        args: the arguments from the command line (beside the program name) that
        should be paths to the raw data files for each genre.
    Returns:
        [genre_names, to_nlp_repeated, to_nlp_unique]: list of the genre names and
        two lists of lists that organize the training and testing data for both
        unique and repeated data variations.
    """
    if len(args) != len(set(args)): #if the arguments are not unique
        print("please enter file paths for different genres")
        sys.exit(0) #to terminate the program
        return
    train_set_size = 60
    test_set_size = 15
    files = []
    genre_names = []
    for arg in args:
        path = Path(arg)
        genre_name = p.basename(path).replace('.tsv', '')
        genre = (genre_name, path)
        files.append(genre)

    #lists inside the tuples
    training_data_repeated = []
    training_data_unique = []

    testing_data_repeated = []
    testing_data_unique = []

    repeated_to_interleave = []
    unique_to_interleave = []

    test_repeated =[]
    test_unique = []

    for file_num, file in enumerate(files):
        genre_name = file[0]
        genre_names.append(genre_name)
        to_train_repeated = []
        to_train_unique = []
        to_test_repeated =[]
        to_test_unique =[]
        with open(file[1]) as file_name:
            for i in range(train_set_size):
                doc_train = file_name.readline()
                doc_repeated = clean(doc_train)
                doc_unique = unique(doc_repeated)

                training_data_repeated = (genre_name, doc_repeated)
                training_data_unique = (genre_name, doc_unique)

                to_train_repeated.append(training_data_repeated)
                to_train_unique.append(training_data_unique)

            repeated_to_interleave.append(to_train_repeated)
            unique_to_interleave.append(to_train_unique)

            for i in range(test_set_size):
                doc_test = file_name.readline()
                doc_repeated = clean(doc_test)
                doc_unique = unique(doc_repeated)

                testing_data_repeated = (genre_name, doc_repeated)
                testing_data_unique = (genre_name, doc_unique)

                to_test_repeated.append(testing_data_repeated)
                to_test_unique.append(testing_data_unique)

            test_repeated.append(to_test_repeated)
            test_unique.append(to_test_unique)

    #interleaving the training data
    interleaved_train_repeated = [element for data in zip(*repeated_to_interleave) for element in data] #the idea of using zip to interleave was inspired from: https://www.xspdf.com/resolution/50312305.html
    interleaved_train_unique = [element for data in zip(*unique_to_interleave) for element in data]
    #interleaving the test data (not needed but why not :D)
    interleaved_test_repeated = [element for data in zip(*test_repeated) for element in data]
    interleaved_test_unique = [element for data in zip(*test_unique) for element in data]

    to_nlp_repeated = [interleaved_train_repeated, interleaved_test_repeated]
    to_nlp_unique = [interleaved_train_unique, interleaved_test_unique]
    return [genre_names, to_nlp_repeated, to_nlp_unique]


def clean(data):
    """ clean data by removing proper nouns and punctuation.
    Args:
        data: a big 1 string of raw data
    Returns:
        data: list of strings (words) with no proper nouns nor punctuation.
    """
    data = [word.split("|", 1)[0].lower() for word in data.split(' ') if "NP" not in word] #remove proper nouns
    data = [word.rstrip(string.punctuation) for word in data] #remove punctuation
    data = list(filter(None, data)) #remove empty strings(they can result after removing punctuation from a string)
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
    """ Retrieve the top 10 words associates with a genre
    Args:
        lex: polarity lexicon (dictionary) that maps a word with its polarity value.
    Returns:
        table: table that hold the top 10 words (with the highest polarity) along
        with their polatity values.
    """
    counter = Counter(lex)
    top10 = counter.most_common(10) #from python3 Collections documentation: https://docs.python.org/3.7/library/collections.html
    values = [[word, polarity] for word, polarity in top10]
    head = ['Top 10 words', 'Polarity']
    table = tabulate(values, head, tablefmt="simple") #from https://pypi.org/project/tabulate/
    return table


def main():
    genres, to_nlp_repeated, to_nlp_unique = preprocess(sys.argv[1:])
    print("Processing the following genres: ",genres)
    training_data_repeated = to_nlp_repeated[0]
    testing_data_repeated = to_nlp_repeated[1]

    training_data_unique = to_nlp_unique[0]
    testing_data_unique = to_nlp_unique[1]

    print("\n training the model with ", len(training_data_repeated), " example documents (with all word occurrences)... \n")
    submodels_repeated = train(genres, training_data_repeated, 10)
    print("\n training the model with ", len(training_data_unique), " example documents (with unique word occurrences)... \n")
    submodels_unique = train(genres, training_data_unique, 10)

    print("\n Confusion Matrix (repeated word occurrences): \n", test(submodels_repeated, testing_data_repeated), "\n")
    for submodel in submodels_repeated:
        print("The top 10 words associated with (",submodel[0],") genre: \n",top_10(submodel[2]) , "\n")
    print("\n Confusion Matrix (unique word occurrences): \n", test(submodels_unique, testing_data_unique), "\n")
    for submodel in submodels_unique:
        print("The top 10 words associated with (",submodel[0],") genre: \n",top_10(submodel[2]), "\n")


if __name__=='__main__':
    main();
