import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S
AP -> Adj | Adj AP
PP -> P NP | P PP
NP -> Det NP | AP NP | N | NP PP NP
VP -> V | V NP | VP Conj VP | VP P NP | VP Adv | Adv VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Import punctuation
    nltk.download('punkt')
    # Split all of the words
    words = nltk.word_tokenize(sentence)
    removing = []
    for i in range(len(words)):
        words[i] =words[i].lower()
        # Now check the number of alphabets
        nums_of_letter = 0
        for letter in words[i]:
            if letter.isalpha():
                nums_of_letter += 1
        # to be removed if num_of_letter < 1
        if nums_of_letter < 1:
            removing.append(i)

    # remove the items that have index in removing
    # starting from back so that the index don't mess up
    for ind in removing[::-1]:
        words.pop(ind)

    return words



def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []

    # For every subtree including itself, I check if its label is NP
    # Then, I check if the other subtrees contain any NP
    # If not, then append the subtree into chunks
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            # Excluding NP that is only N
            if len(subtree.leaves()) == 1:
                continue
            if not check_subtree_contain_np(subtree):
                chunks.append(subtree)

    return chunks

# A helper function to check if subtrees contain NP label
# True if contains false if doesn't
def check_subtree_contain_np(tree):
    for subtree in tree.subtrees(lambda x: x != tree):
        if subtree.label() == "NP":
            # Check for NP that is singly N
            # ex: NP -> N -> paint
            # Then it is not considered as a NP
            if len(subtree.leaves()) != 1:
                return True

    return False


if __name__ == "__main__":
    main()
