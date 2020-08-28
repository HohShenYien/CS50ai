import sys
import time
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # First I'll check for each variables
        for var in self.domains:
            # Create a new domain for each variable that has correct word length
            new_dom = []
            for word in self.domains[var]:
                if var.length == len(word):
                    new_dom.append(word)
            if len(self.domains[var]) != len(new_dom):
                self.domains[var] = new_dom
                
    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # Get the overlaps        
        overlaps = self.crossword.overlaps[x, y]
        modified = False
        posX, posY = overlaps
        
        # Now check the words
        # See if the overlap position has the same letter for words in x and y domains
        # If words in x domains has same letter with any word in y domains at the particular position,
        # Then the word is okay
        removed = []
        for words_x in self.domains[x]:
            # Check if arc consistent
            consistent = False
            for words_y in self.domains[y]:
                if words_x[posX] == words_y[posY]:
                    consistent = True
                    break
            
            # Remove the word if not consistent, and indicate that a change is made    
            if not consistent:
                removed.append(words_x)    
                modified = True           
                
        # Take out the word 
        if modified:
            for word in removed:
                self.domains[x].remove(word)
        return modified

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = []
            # Get every pair which has intersect
            for var1 in self.domains:
                for var2 in self.domains:
                    if var1 == var2:
                        continue
                    
                    if self.crossword.overlaps[var1, var2] is not None:
                        arcs.append((var1, var2))
        # Check for every pair in the list                
        while len(arcs) != 0:
            (X, Y) = arcs.pop(0)
            if self.revise(X, Y):
                # Check if the domains is empty or not
                if len(self.domains[X]) == 0:
                    return False
                
                else:
                    for neighbor in self.crossword.neighbors(X) - {Y}:
                        arcs.append((X, neighbor))
                        
        return True
            

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Only True if all variables are in assignment 
        for var in self.domains:
            if var in assignment and assignment[var]:
                continue
            return False
        
        return True
            

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        
        distinct = set()
        for var in assignment:
            # Check for corrent word length
            if len(assignment[var]) != var.length:
                return False
            
            # Check for conflicts
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    continue
                # Get the overlaps
                overlaps = self.crossword.overlaps[var, neighbor]
                posX, posY = overlaps
                if assignment[var][posX] != assignment[neighbor][posY]:
                    return False
            distinct.add(assignment[var])
            
        # Check for distinct
        if len(distinct) != len(assignment):
            return False
        
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        words = set(self.domains[var])
        # First, I'll remove the words that are already assigned
        for word in assignment.values():
            if word in words:
                words.remove(word)
        # Make a simple helper function that counts number
        # of neighboring variables the word rules out
        def helper(word):
            val = 0
            for variable in self.crossword.neighbors(var):
                if word in self.domains[variable]:
                    val += 1
            
            return val
        
        words = list(words)
        # Now I can sort it
        words.sort(key=helper)    
        return words

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Get the unassigned first
        unassigned = [var for var in self.domains if var not in assignment]
        # Sort
        unassigned.sort(key = lambda x: len(self.crossword.neighbors(x)))
        return unassigned.pop()

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        
        for word in self.order_domain_values(var, assignment):
            assignment[var] = word
            #old_domain = self.domains.copy()
            if self.consistent(assignment):
                # My inference, somehow it makes the program slower
                '''
                self.ac3([(var, neighbor) for neighbor in self.crossword.neighbors(var)])
                
                for var in self.domains:
                    if len(self.domains[var]) == 1:
                        assignment[var] = self.domains[var][0]
                        '''
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                
            assignment.pop(var)
            #self.domains = old_domain
            
        return None

def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    now = time.time()
    for i in range(20):
        main()
    print((time.time() - now) / 20)
