# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses functions that interface with either Gensim or Fasttext
libraries for training word embeddings.
"""

import subprocess
import joblib
import fasttext
import gensim
from . import notifiers


@notifiers.do_cprofile
def fasttext_model(input_data, filename):
    """ Train a fasttext model
    """
    cpu_count = joblib.cpu_count()
    _model = fasttext.skipgram(input_data, filename, thread=cpu_count)


@notifiers.do_cprofile
def word2vec_model(input_data, filename):
    """ Train a word2vec model
    """
    cpu_count = joblib.cpu_count()
    sentences = gensim.models.word2vec.LineSentence(input_data)
    model = gensim.models.Word2Vec(sentences, min_count=5, workers=cpu_count)
    model.save(filename)


# @notifiers.do_cprofile
def fasttext_classifier(input_data, filename, lr=0.1, dim=100, ws=5, epoch=5, min_count=1, word_ngrams=1):
    """ Train a fasttext model
    See https://github.com/salestock/fastText.py for params.
    """
    cpu_count = joblib.cpu_count()
    _classifier = fasttext.supervised(
        input_data, filename, thread=cpu_count, lr=lr, dim=dim, ws=5, epoch=epoch, min_count=1, word_ngrams=1)


def dep2vec_model(filename, filter_count, min_count, dimensions):
    """ Train a dependency2vec model.
    Follows the same steps outlined in the dependency2vec readme
    """
    time1 = notifiers.time()
    # Step 1
    output = subprocess.run("cut -f 2 hatespeech_core/data/conll_data/" + filename + " | python hatespeech_core/data/conll_data/dependency2vec/scripts/vocab.py " + str(
        filter_count) + " > " + "hatespeech_core/data/conll_data/dependency2vec/vocab_data/counted_vocabulary_" + filename, shell=True, check=True, stdout=subprocess.PIPE)
    for line in output.stdout.splitlines():
        print(line)
    print(output)

    # Step 2
    output = subprocess.run("cat hatespeech_core/data/conll_data/" + filename + " | python hatespeech_core/data/conll_data/dependency2vec/scripts/extract_deps.py hatespeech_core/data/conll_data/dependency2vec/vocab_data/counted_vocabulary_" +
                            filename + " " + str(filter_count) + " > " + "hatespeech_core/data/conll_data/dependency2vec/vocab_data/dep.contexts_" + filename, shell=True, check=True, stdout=subprocess.PIPE)
    for line in output.stdout.splitlines():
        print(line)

    # Step 3
    output = subprocess.run("hatespeech_core/data/conll_data/dependency2vec/" + "./count_and_filter -train " + "hatespeech_core/data/conll_data/dependency2vec/vocab_data/dep.contexts_" + filename + " -cvocab " +
                            "hatespeech_core/data/conll_data/dependency2vec/vocab_data/cv_" + filename + " -wvocab " + "hatespeech_core/data/conll_data/dependency2vec/vocab_data/wv_" + filename + " -min-count " + str(min_count), shell=True, check=True, stdout=subprocess.PIPE)
    for line in output.stdout.splitlines():
        print(line)

    # Step 4
    output = subprocess.run("hatespeech_core/data/conll_data/dependency2vec/" + "./word2vecf -train " + "hatespeech_core/data/conll_data/dependency2vec/vocab_data/dep.contexts_" + filename + " -cvocab " +
                            "hatespeech_core/data/conll_data/dependency2vec/vocab_data/cv_" + filename + " -wvocab " + "hatespeech_core/data/conll_data/dependency2vec/vocab_data/wv_" + filename + " -size " + str(dimensions) + " -negative 15 -threads 10 -output hatespeech_core/data/persistence/word_embeddings/dim" + str(dimensions) + "vecs_" + filename, shell=True, check=True, stdout=subprocess.PIPE)
    for line in output.stdout.splitlines():
        print(line)

    time2 = notifiers.time()
    notifiers.send_job_completion(
        [time1, time2], ["dependency2vec", "dependency2vec " + filename])
