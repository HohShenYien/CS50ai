import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000
SENSITIVITY = 0.001


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Declare a dictionary with 0 probability first
    probs = {x: 0 for x in corpus}
    pages = len(corpus)
    # Consider all the pages under the page
    for sub_page in corpus[page]:
        ## Adding up their probability
        probs[sub_page] += damping_factor / len(corpus[page])

    # Check for other pages
    for next_page in corpus:
        # Exclude the given page
        if next_page == page:
            continue

        for sub_page in corpus[next_page]:
            # Adding up their probability
            probs[sub_page] += (1 - damping_factor) / len(corpus[next_page]) / (pages - 1)

    return probs



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Declare a dictionary of all pages with 0 rank
    sample = {page: 0 for page in corpus.keys()}

    # Pick randomly for the first time
    cur_page = random.choice(list(corpus.keys()))
    sample[cur_page] += 1

    # Now repeat for n - 1 times to get the full sample
    for i in range(n - 1):
        # 85% picking from links
        if random.random() <= 0.85:
            # Choices to select
            choices = transition_model(corpus, cur_page, damping_factor)
            # Get the cumulative sum of probability and pick the page
            # If the random generator returns a number less than the page's
            # cumsum.
            cumsum = list(accumu(choices.values()))
            dice = random.random()
            for i in range(len(choices)):
                if dice <= cumsum[i]:
                    cur_page = list(choices.keys())[i]
                    break

        else:
            cur_page = random.choice(list(corpus.keys()))

        sample[cur_page] += 1

    # Now divide the number of sample times with length to get ratio
    total = 0
    for page in sample:
        total += sample[page]
    for page in sample:
        sample[page] /= total

    return sample

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Declaring the starting dictionary
    rank_old = {page: 1 / len(corpus) for page in corpus.keys()}

    while True:
        # Empty dictionary to be filled
        rank_new = {page: 0 for page in corpus.keys()}

        for page in rank_new:
            for link in corpus[page]:
                # Adding up rank from each page
                rank_new[link] += (rank_old[page] / len(corpus[page]))

        for page in rank_new:
            # Processing the remaining
            rank_new[page] *= damping_factor
            rank_new[page] += (1 - damping_factor) / len(corpus)

        # If converge then return
        if check_accuracy(rank_old, rank_new):
            return rank_new
            break
        rank_old = rank_new



# A simple function for cumulutive sum
def accumu(li):
    total = 0
    for prob in li:
        total += prob
        yield total

# A simple helper function to help me check for convergence
def check_accuracy(prev, new):
    for page in prev:
        if abs(prev[page] - new[page]) >= SENSITIVITY:
            return False

    return True

if __name__ == "__main__":
    main()
