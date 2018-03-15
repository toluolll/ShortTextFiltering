import pickle
import numpy
import math
import heapq
import sys
import random


def stem(token):
    if len(token) < 4:
        return token
    if token.endswith("ing"):
        token = token[:-3]
    elif token.endswith("ed"):
        token = token[:-2]
    elif token.endswith("es"):
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3 and token[-2] in "wrtpsdfgklmnbvcz":
        token = token[:-1]
    elif token.endswith("e") or token.endswith("y"):
        token = token[:-1]
    return token

def norm(vec):
  return numpy.array(vec) / math.sqrt(sum(val * val for val in vec))

def to_vec(line):
    chunks = line.rstrip().split(" ")
    if len(chunks) != 301:
        return []
    word = chunks[0]
    vector = norm([float(val) for val in chunks[1:]])
    return [(word, vector)]

def cosine(first, second):
    return (1 - sum(first * second)) 

def find_shortest_transform(first_w2v, second_w2v, word_dist_metric=cosine):
    from scipy.optimize import linprog
    costs = [word_dist_metric(first_w2v[first], second_w2v[second])  for first in xrange(len(first_w2v)) for second in xrange(len(second_w2v))]
    first_word_weight = 1.0 / len(first_w2v)
    second_word_weight = 1.0 / len(second_w2v)
    eq_restrictions = []
    for first in xrange(len(first_w2v)):
        eq_restrictions += [[0.0] * (first * len(second_w2v)) + [1.0] * len(second_w2v) + [0.0] *  ((len(first_w2v) - first - 1) * len(second_w2v))]
    for second in xrange(len(second_w2v)):
        vec = [0.0] * (len(first_w2v) * len(second_w2v))
        for first in xrange(len(first_w2v)):
            vec[first * len(second_w2v) + second] = first_word_weight
        eq_restrictions += [vec]
    eq_restrictions = numpy.matrix(eq_restrictions) #.transpose()
    eq_vals = [1] * len(first_w2v) + [second_word_weight] * len(second_w2v)
    lims = [(0, 1.0)] * (len(first_w2v) * len(second_w2v))
    res = linprog(costs, A_eq=eq_restrictions, b_eq=eq_vals, bounds=lims,  options={"disp": False})
    distance = sum(res.x * costs)
    return distance / float(len(first_w2v))


