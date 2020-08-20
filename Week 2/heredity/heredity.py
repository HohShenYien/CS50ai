import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Since the final probability involves product of all probability
    # So initial is 1
    probs = 1;
    
    # Iterate through all the people
    for person in people:
        cur_prob = 0
        # If the person has parent, then calculate from their parent
        if have_parent(people, person):
            father = people[person]['father']
            mother = people[person]['mother']
            # Check the probability that the person has 1 gene
            if person in one_gene:
                # Get from father but not mother
                cur_prob += inherit(number_of_gene(father, one_gene, two_genes)
                                    , 1) * inherit(number_of_gene(mother, one_gene, two_genes)
                                    , 0)
                                    
                # Get from mother but not father
                cur_prob += inherit(number_of_gene(mother, one_gene, two_genes)
                                    , 1) * inherit(number_of_gene(father, one_gene, two_genes)
                                    , 0)
            # Check for 2 genes                        
            elif person in two_genes:
                # Get from both parent
                cur_prob += inherit(number_of_gene(father, one_gene, two_genes), 1) *\
                            inherit(number_of_gene(mother, one_gene, two_genes), 1)
                         
            # And 0 gene   
            else:
                # Get none from both parent
                cur_prob += inherit(number_of_gene(father, one_gene, two_genes), 0) *\
                            inherit(number_of_gene(mother, one_gene, two_genes), 0)
        
        # If parent not stated, then get from default probability                    
        else:
            cur_prob += PROBS['gene'][number_of_gene(person, one_gene, two_genes)]
            
        # Finally compute if the person have trait or not
        cur_prob *= PROBS['trait'][number_of_gene(person, one_gene, two_genes)][person in have_trait]
        
        # Then multiply the cumulative probs with cur_prob
        probs *= cur_prob
        
    # Return the cumulative probs
    return probs
                



def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Iterate for every person
    for person in probabilities:
        # Just check the number of genes they have and add the probs p
        probabilities[person]['gene'][number_of_gene(person, one_gene, two_genes)] += p
        
        # Then add the have_trait or not
        probabilities[person]['trait'][person in have_trait] += p
        


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Just a helper generator to reduce codes
    genes = range(3)
    
    # Iterate for every person
    for person in probabilities:
        # Add up all probabilities in gene and divide
        gene_total = sum(probabilities[person]['gene'].values())
        for gene in genes:
            probabilities[person]['gene'][gene] /= gene_total
            
        # Similarly for trait
        trait_total = sum(probabilities[person]['trait'].values())
        probabilities[person]['trait'][True] /= trait_total
        probabilities[person]['trait'][False] /= trait_total

def have_parent(people, person):
    return people[person]['father'] is not None

# A helper function to find the probability to inherit 
# number of gene from parent given parent has this much of gene
def inherit(parent_number, number_gene):
    if parent_number == 2:
        if number_gene == 1:
            return 1 - PROBS['mutation']
        else:
            return PROBS['mutation']
    
    elif parent_number == 1:
        return 0.5
    
    elif parent_number == 0:
        if number_gene == 1:
            return PROBS['mutation']
        else:
            return 1 - PROBS['mutation']

# Another helper function to help calculate number of gene this person have
def number_of_gene(person, one_gene, two_genes):
    if person in one_gene:
        return 1
    elif person in two_genes:
        return 2
    else:
        return 0

if __name__ == "__main__":
    main()
