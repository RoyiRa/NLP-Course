from collections import defaultdict
import random
from docopt import docopt
import sys


class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l,r,w)
        return grammar

    def is_terminal(self, symbol): return symbol not in self._rules

    def gen(self, symbol, print_tree):
        if self.is_terminal(symbol):
            return symbol
        else:
            expansion = self.random_expansion(symbol)
            output = ' '.join([self.gen(symbol=s, print_tree=print_tree) for s in expansion])
            return f"({symbol} {output})" if print_tree else output

    def random_sent(self, print_tree):
        sen = self.gen("ROOT", print_tree=print_tree)
        return sen

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r


if __name__ == '__main__':
    doc = """Usage: generate.py FILE [-htn COUNT] 
    
        -n COUNT     specify how many sentences to generate [default: 1]
        -t           specify if to generate sentence tree structure [default: False]
                    
"""
    arguments = docopt(doc, help=True, version=None, options_first=False)
    grammar_file = arguments['FILE']
    N = int(arguments['-n'])
    t = arguments['-t']

    for i in range(N):
        pcfg = PCFG.from_file(grammar_file)
        print(pcfg.random_sent(print_tree=t))