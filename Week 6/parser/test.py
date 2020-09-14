import nltk

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
S -> NP VP | N VP | S Conj S
NP -> Det NP | NP P NP | Adj NP | Adj N | N P N | Det N
VP -> V | V NP | VP Conj VP | VP P NP | VP Adv | V N | VP P N | Adv VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

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
        # to be removed if num_of+letter < 1
        if nums_of_letter < 1:
            removing.append(i)

    # remove the items that have index in removing
    # starting from back so that the index don't mess up
    for ind in removing[::-1]:
        words.pop(ind)

    return words

sentence = "She never said a word until we were at the door here."
s = preprocess(sentence)
print(s)
trees = list(parser.parse(s))
for tree in trees:
    tree.pretty_print()
    print(tree.leaves())
