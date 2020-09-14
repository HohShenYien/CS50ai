import nltk
import sys
import os
from string import punctuation
from math import log

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = {}

    for path, subdirs, files in os.walk("corpus"):
        for file in files:
            # Confirm it ends with .txt
            if file[-4:] != ".txt":
                continue
            # Adds into dictionary
            with open(os.path.join(directory, file), "r",encoding="utf8") as f:
                corpus[file] = f.read()

    return corpus

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # Split all of the words
    words = nltk.word_tokenize(document)

    # Removing punctuations and common words
    removing = []
    for i in range(len(words)):
        words[i] = words[i].lower()

        # Remove any punctuation
        words[i] = "".join(list(filter(lambda x: x not in punctuation, words[i])))

        # Make sure it's not common words
        if words[i] in nltk.corpus.stopwords.words("english"):
            removing.append(i)
            continue

        # to be removed if number of letter < 1
        if len(words[i]) < 1:
            removing.append(i)
            continue

    # remove the items that have index in removing
    # starting from back so that the index don't mess up
    for ind in removing[::-1]:
        words.pop(ind)

    # return in order
    return sorted(words)

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Using an extra dictionary that maps names of documents to a set
    # to improve efficiency a bit
    tmp = {file: set(documents[file]) for file in documents}
    words = {}
    for file in tmp:
        for word in tmp[file]:
            # No need do repetition
            if word in words:
                continue
            # Computing the log
            words[word] = log(len(documents) / len([x for x in tmp if word in tmp[x]]))

    return words

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # I will use a dictionary instead of using count method directly because
    # more efficient
    # words dictionary
    words = {word:0 for word in query}

    # Compute total frequency of  in query first
    tf = {file:words.copy() for file in files}
    for file in files:
        for word in files[file]:
            if word in query:
                tf[file][word] += 1
    # Compute the tfidf values for all files
    # A list of (filename, tfidf value)
    tfidfs = []
    for file in tf:
        value = 0
        for word in tf[file]:
            value += tf[file][word] * idfs[word]

        tfidfs.append((file, value))

    tfidfs.sort(key=lambda x: x[1],reverse=True)

    return [x[0] for x in tfidfs[:n]]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Using count doens't spend too much time here because all the sentences are short
    res = []
    for sentence in sentences:
        idf = 0
        words_in_query = 0
        for word in query:
            if word in sentences[sentence]:
                words_in_query += 1
                idf += idfs[word]

        density = words_in_query / len(sentences[sentence])
        res.append((sentence, idf, density))

    res.sort(key = lambda x: (x[1], x[2]),reverse=True)
    # Sort and return
    return [s[0] for s in res[:n]]


if __name__ == "__main__":
    main()
