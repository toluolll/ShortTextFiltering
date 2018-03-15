# ShortTextFiltering
This repository aggregates some code and anonymized tweets dataset used for the topical filtering of short texts. More details can be found in the CIKM'17 paper "Efficient Document Filtering Using Vector Space Topic Expansion and Pattern-Mining: The Case of Event Detection in Microposts".

exeriments.sh has the final commends to run the processing and the extraction.
submit_spark.sh - contains the cluster configuration required to run the extraction.
build_patterns.py - is an actual implementation and stats computation of the algorithm.

NOTE: texts of the tweets were removed from the release (according to the Twitter policy), however, tweet ids are still preserved.
As a result the following files should be generated prior to using the code:

twitter_unigrams.txt - is derived from the corpus of tweets that is available to you. We have used the 5% dump of the 2014, 2015 year.
This file contains "token\tdocument_requency".

index_word_groups_30001.txt - contains words synsets and is available in the repo.

matching_attacks_and_tweets_from_ter_db_filtered.txt - contains the information between atacks description from GTD or wikipedia
as well as matched words in the tweets. In the end of each line, corresponding tweet id is added.
IMPORTANT: you will need to attach the texts of the tweets (by provided tweet_id) to each line of the file.

wiki.en.vec.500K - contains the wordembeddings trained on wiki and are publicly available.
