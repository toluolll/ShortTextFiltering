import pickle
import numpy
import math
import heapq
import sys
import random
import subprocess
import json
from utils import *
from pyspark import SparkContext


stop_words = []
max_freq = 0.005 * 60000000
# A list of \t separated unigrams and their corresponding counts in the whole corpus.
for line in open("twitter_unigrams.txt"):
  word, freq = line.decode("utf8").split("\t")
  if float(freq) > max_freq:
    stop_words.append(word)

stop_words = set(stop_words)

# Map of words and their corresponding group name, e.g., {"a": gourp1, "the": group1}.
# It is available in the repo.
index_word_groups_filename = "index_word_groups_30001.txt"
word_group_index = json.loads(open(index_word_groups_filename, "r").read())

def word2data(tweet_data):
    id, tokens_str = tweet_data.split("\t")
    tokens = set(tokens_str.split(" "))
    selected =  [token for token in tokens if token in wiki_words and not stem(token) in stop_words]
    if len(selected) > 3 and len(selected) > 0.5 * len(tokens):
        sel_str  = " ".join(selected)
        for token in selected:
            yield (token, (id, tokens_str + "|" + sel_str ))

def serialize(tweet_data):
    (id, text), vec = tweet_data
    vec = norm(vec)
    return id + "\t" + text + "\t" + " ".join(str(val) for val in vec)


def rec(stack, sel_docs_sorted, d2c, max_length, output):
  if len(stack) > 1:
    output.append( (set(stack), len(sel_docs_sorted)) )
  if len(stack) >= max_length:
    return
  c2d = {}
  for doc in sel_docs_sorted:
    for cat in d2c[doc]:
      c2d.setdefault(cat, []).append(doc)    
  min_key = stack and stack[-1] or "" 
  all_cats = c2d.keys()
  all_cats.sort()
  for cat in all_cats:
    if cat <= min_key:
      continue
    cat_docs = c2d[cat]
    rec(stack + [cat], cat_docs, d2c, max_length, output) 

sc = SparkContext()

# "\t" separated matched twets info:
#   attack id,
#   tfidf score,
#   matched words between wiki description and tweet,
#   wiki attack description,
#   tweet_id,
#   tweet_text.

# Attacks desciptions and further information is available in attacksUpdate.json in the repo.
# tweet_text should be appended once the texts are downloaded.
# We can not release the actual tweet texts due to the Twitter policy.
attack_tweets_file = "matching_attacks_and_tweets_from_ter_db_filtered.txt"
def extract_ids(line):
    chunks = line.split("\t")
    if len(chunks) == 6:
        yield chunks[-2]

# Per line: tweet_id\tlist of words/token (without repetition).
# It will be around 4Gb when the corresponding (that are available in the repo)
# tweet ids will be downloaded.
tweets_folder = "eng_tweets2tokens_no_rep.txt"

selected_ids = set(sc.textFile(attack_tweets_file).flatMap(extract_ids).collect())
seen = sc.textFile(tweets_folder).filter(lambda line: line.split("\t")[0] in selected_ids).collect()
selected_ids = set([line.split("\t")[0] for line in seen])
pickle.dump(selected_ids, open("attack_ids.wb", "wb"))
print "training ids", len(selected_ids)
# selected_ids = pickle.load(open("attack_ids.wb", "rb"))

training_ids = list(selected_ids)
random.shuffle(training_ids)
TRAINING_SIZE = int(sys.argv[1])
MAX_COMB_LEN = int(sys.argv[2])
print "training size", TRAINING_SIZE
print "max_comb_len", MAX_COMB_LEN

training_ids = set(training_ids[:TRAINING_SIZE])

test_ids = set(selected_ids) - training_ids
print "test_ids", len(test_ids)

training_texts = sc.textFile(tweets_folder).filter(lambda line: line.split("\t")[0] in training_ids).collect()

all_words = []
id2words = {}

id = 0
for line in training_texts:
  words = set([stem(word.strip()) for word in line.split("\t")[-1].strip().split(" ") if not stem(word.strip()) in stop_words])
  if len(words) < 3:
    continue
  id2words[id] = list(words)
  id += 1

print len(training_texts), "docs", len(id2words)
sel_docs_sorted = id2words.keys()
sel_docs_sorted.sort()
pre_combs = []
rec([], sel_docs_sorted, id2words, 2,  pre_combs)

print "pre_combs", len(pre_combs)
key2combs = {}
for c_index in xrange(len(pre_combs)):
  comb, loc_freq = pre_combs[c_index]
  first_key = min(comb)
  key2combs.setdefault(first_key, []).append(c_index)
max_freq = max((len(indices), key) for key, indices in key2combs.items())
print "max number of pre_combs assoc with key", max_freq

def combs_tweet_samples_select(tweet_line):
  id, tokens = tweet_line.split("\t")
  tokens = [word.strip() for word in tokens.strip().split(" ") if not stem(word.strip()) in stop_words]
  tokens += [word_group_index[word] for word in tokens if word in word_group_index]
  stemmed = set([stem(token) for token in tokens]) 
  if len(stemmed) >= 3:
    for token in stemmed:
      if token in key2combs:
        for comb_index in key2combs[token]:
          if not (pre_combs[comb_index][0] - stemmed):
            yield (comb_index, [(id, tokens)])

def sample_docs(elems):
  SAMPLE_SIZE = 100
  if len(elems) <= SAMPLE_SIZE:
    return elems
  else:
    import random
    return random.sample(elems, SAMPLE_SIZE)

def rec_yield(stack, sel_docs_sorted, d2c, special, max_length):
  if len(sel_docs_sorted) > 1:
    yield ("|".join(stack), len(sel_docs_sorted), len(special), sample_docs(sel_docs_sorted))
    if len(stack) < max_length:
      c2d = {}
      for doc in sel_docs_sorted:
        for cat in d2c[doc]:
          c2d.setdefault(cat, []).append(doc)    
      min_key = stack and stack[-1] or "" 
      all_cats = c2d.keys()
      all_cats.sort()
      for cat in all_cats:
        if cat <= min_key:
          continue
        cat_docs = c2d[cat]
        new_special = special & set(cat_docs)
        if new_special:
          for return_value in rec_yield(stack + [cat], cat_docs, d2c, new_special, max_length):
            yield return_value

def grow(comb_index2docs):
  comb_index, docs = comb_index2docs
  stack = sorted(list(pre_combs[comb_index][0]))
  d2c = {}
  special = set()
  id2tokens = {}
  for doc_id, tokens in docs:
    id2tokens[doc_id] = tokens
    keys = sorted(list(set([stem(token) for token in tokens])))
    d2c[doc_id] = keys
    if doc_id in training_ids:
      special.add(doc_id)
  for comb, freq, local_freq, sample_ids in rec_yield(stack, sorted(d2c.keys()), d2c, special, MAX_COMB_LEN):
    for id in sample_ids:
      for token in id2tokens[id]:
        yield (token, (comb, freq, local_freq, id)) 
        

docs_aggreg_by_precombs = sc.textFile(tweets_folder).flatMap(combs_tweet_samples_select).reduceByKey(lambda a, b: a + b)
print "start growing!!"
token2data = docs_aggreg_by_precombs.flatMap(grow)
# Word embeddings trained on wiki. Publicly available.
word2vec = sc.textFile("wiki.en.vec.500K").repartition(1000).flatMap(to_vec)
comb2samples = token2data.join(word2vec).map(lambda (token, ((comb, freq, local_freq, id), vec )) : ((comb, freq, local_freq), [(id, token, vec)]) ).reduceByKey(lambda a,b:a+b)

#for (comb, freq, local_freq), data in comb2samples.sample(False, 0.0001).collect()[:10]:
#  print "\tsample", comb, "total_freq:", freq, "loc_freq", local_freq, "docs:", len(set(id for id, _, _ in data))

print "initial combs size", comb2samples.count()

def get_distances_and_filter(key_val):
  (comb, freq, local_freq), docs_vecs = key_val
  by_docs = {}
  for id, token, vec in docs_vecs:
    by_docs.setdefault(id, []).append(vec)
  docs = by_docs.values()
  pairs = [(first, second) for first in xrange(len(docs)) for second in xrange(first + 1, len(docs))]
  sample_size = 30
  sample = len(pairs) > sample_size and random.sample(pairs, sample_size) or pairs
  distances = [find_shortest_transform(docs[first], docs[second]) for first, second in sample]
  import numpy
  if distances and numpy.mean(distances) <= 0.48 and local_freq > 1:
    yield (comb, freq, local_freq, numpy.median(distances))


selected_combs = comb2samples.flatMap(get_distances_and_filter).collect()
print "selected", len(selected_combs)
combs = [(set(comb.split("|")), comb, freq, local_freq, median) for comb, freq, local_freq, median in selected_combs]

for data in combs[:50]:
  print "comb sample\t", data

key2combs = {}
for index in xrange(len(combs)):
  min_key = min(combs[index][0])
  key2combs.setdefault(min_key, []).append(index)

def extract_matching_tweets(tweet_line):
  id, tokens = tweet_line.split("\t")
  tokens = [ word for word in tokens.strip().split(" ") ]
  tokens += [word_group_index[word] for word in tokens if word in word_group_index]
  stemmed = set([stem(word.strip()) for word in tokens])
  matched = []
  for token in stemmed:
    if token in key2combs:
      for comb_index in key2combs[token]:
        if not (combs[comb_index][0] - stemmed):
          matched += [comb_index]
  if matched:
    yield (matched, tweet_line)
  
 
extracted_cloud = sc.textFile(tweets_folder).flatMap(extract_matching_tweets)
print "test size:", len(test_ids)
count_extracted = extracted_cloud.filter(lambda line: line[1].split("\t")[0] in test_ids).count()
print "extracted for recall:", count_extracted
recall = float(count_extracted) / len(test_ids)
import random
print "QUAL", MAX_COMB_LEN, TRAINING_SIZE
print "total extracted", extracted_cloud.count(), "recall", recall
extracted = extracted_cloud.collect()
from collections import defaultdict
pattern_to_tweet = defaultdict(list)
for matched, tweet_line in random.sample(extracted, 50):
  print "matched", len(matched)
  for comb_index in matched[:100]:
    print combs[comb_index]
    pattern_to_tweet[ "_".join([str(e) for e  in combs[comb_index][1:]] ) ] += [ tweet_line.encode("utf8") ]
  print "text"
  print "\t", tweet_line.encode("utf8")
print "-----"

print "number of patterns: ", len(pattern_to_tweet)
for k, v in pattern_to_tweet.iteritems():
  print k
  print v

