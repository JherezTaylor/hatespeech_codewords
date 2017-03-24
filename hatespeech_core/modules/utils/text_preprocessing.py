# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Handles text preprocessing
"""

import re
import string
import copy
from itertools import chain
from joblib import Parallel, delayed, cpu_count
from textblob import TextBlob
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from pymongo import UpdateOne
# from nltk.sentiment.util import demo_sent_subjectivity as SUBJ
from . import EmotionDetection
from . import twokenize
from . import notifiers

PUNCTUATION = list(string.punctuation)
STOP_LIST = set(stopwords.words(
    "english") + PUNCTUATION + ["rt", "via", "RT"])


@notifiers.timing
def preprocess_text(raw_text):
    """Preprocessing pipeline for raw text.

    Tokenize and remove stopwords.

    Args:
        raw_text  (str): String to preprocess.

    Returns:
        list: vectorized tweet.
    """

    cleaned_result = clean_tweet_text(raw_text)
    terms_single = cleaned_result[1]

    # unigrams = [ w for doc in documents for w in doc if len(w)==1]
    # bigrams  = [ w for doc in documents for w in doc if len(w)==2]

    hashtags_only = []
    user_mentions_only = []
    terms_only = []

    for token in terms_single:
        if token.startswith("#"):
            hashtags_only.append(token)

        if token.startswith("@"):
            user_mentions_only.append(token)

        if not token.startswith(("#", "@")) and token not in STOP_LIST:
            terms_only.append(token)

    sentiment = TextBlob(raw_text).sentiment
    return {"hashtags": hashtags_only, "user_mentions": user_mentions_only,
            "tokens": terms_only, "sentiment": list(sentiment)}


def get_sent_subj(raw_text):
    """ Return the subjectivity and sentiment scores of the text.

    Args:
        raw_text  (str): String to preprocess.

    Returns:
        list: subjectivity and sentiment scores.
    """

    # Use VADER approach. The compound score is a normalized value of the
    # pos, neg and neu values with -1 being most negative and +1 being most positive.
    # pos, neu and neg are ratios for the proportion of text that fall into
    # each category
    sent_analyzer = SIA()
    sentiment = sent_analyzer.polarity_scores(raw_text)

    # Use TextBlob subjectivity for now
    subjectivity = TextBlob(raw_text).subjectivity
    return [sentiment, subjectivity]


def preprocess_tweet(tweet_obj, hs_keywords, args):
    """Preprocessing pipeline for Tweet body.

    Tokenize and remove stopwords.

    Args:
        tweet_obj   (dict): Tweet to preprocess.
        hs_keywords (set): Keywords to match on.
        args        (list): Contains a list of conditions as follows:
            0: subj_check (bool): Check text for subjectivity.
            1: sent_check (bool): Check text for sentiment.
            2: intensity_measure (bool) Retain tokens with all uppercase.

    Returns:
        dict: Tweet with vectorized text appended.
    """

    # SUBJ is a trained classifer on the movie dataset in NLTK
    # SUBJ(tweet_obj["text"])

    subj_check = args[0]
    sent_check = args[1]

    if subj_check and sent_check:
        subj_sent = get_sent_subj(tweet_obj["text"])
        sentiment = subj_sent[0]
        subjectivity = subj_sent[1]
        # Value between -1 and 1 - TextBlob Polarity explanation in layman's
        # terms: http://planspace.org/20150607-textblob_sentiment/

        # Negative sentiment and possibly subjective
        if sentiment["compound"] < 0 and sentiment["neg"] >= 0.5 and subjectivity >= 0.4:

            processed_text = prepare_text(
                tweet_obj["text"].lower(), [STOP_LIST, hs_keywords])

            tweet_obj = prep_tweet_body(
                tweet_obj, [True, subjectivity, sentiment], processed_text)
            return tweet_obj
        else:
            pass

    if subj_check and not sent_check:
        pass

    if not subj_check and sent_check:
        pass

    elif not subj_check and not sent_check:
        processed_text = prepare_text(
            tweet_obj["text"].lower(), [STOP_LIST, hs_keywords])

        tweet_obj = prep_tweet_body(
            tweet_obj, [False], processed_text)
        return tweet_obj


def prepare_text(raw_text, args):
    """ Clean and tokenize text. Create ngrams.
    Ensure that raw_text is lowercased if that's required.

    Args:
        raw_text  (str): String to preprocess.
        args      (list): Contains the following:
            0: stop_list (list): English stopwords and punctuation.
            1: hs_keywords (list) HS corpus.
            2: intensity_measure (bool) Retain tokens with all uppercase.

    Returns:
        list: terms_only, stopword_only, hashtags, mentions, hs_keyword_count, ngrams,
        stopword_ngrams
    """

    stop_list = args[0]
    hs_keywords = args[1]

    cleaned_result = clean_tweet_text(raw_text)
    clean_text = cleaned_result[0]
    terms_single = cleaned_result[1]

    stopwords_only = set(terms_single).intersection(stop_list)
    terms_only = []
    hashtags = []
    mentions = []

    for token in terms_single:
        if not token.startswith(("#", "@")) and token not in stop_list:
            terms_only.append(token)
        if token.startswith(("#")):
            hashtags.append(token)
        if token.startswith(("@")):
            mentions.append(token)

    xgrams = ([(create_ngrams(clean_text, i)) for i in range(1, 6)])
    stopword_ngrams = copy.deepcopy(xgrams)

    # Filter out ngrams that consist wholly of stopwords
    for i in range(0, 5):
        xgrams[i] = [gram for gram in xgrams[i] if not set(
            twokenize.tokenizeRawTweetText(gram)).issubset(stop_list)]

    # Filter out ngrams that don't consist wholly of stopwords
    for i in range(0, 5):
        stopword_ngrams[i] = [gram for gram in stopword_ngrams[i] if set(
            twokenize.tokenizeRawTweetText(gram)).issubset(stop_list)]

    # Flatten list of lists
    stopword_ngrams = list(chain.from_iterable(stopword_ngrams[1:]))

    # Check the number of HS keyword instances
    grams = list(chain.from_iterable(xgrams))
    hs_keywords_intersect = hs_keywords.intersection(set(grams))

    return [terms_only, list(stopwords_only), hashtags, mentions,
            list(hs_keywords_intersect), xgrams, stopword_ngrams]


def prep_tweet_body(tweet_obj, args, processed_text):
    """ Format the incoming tweet

    Args:
        tweet_obj (dict): Tweet to preprocess.
        args (list): Various datafields to append to the object.
            0: subj_sent_check (bool): Check for subjectivity and sentiment.
            1: subjectivity (num): Subjectivity result.
            2: sentiment (dict): Sentiment result.
        processed_text (list): List of tokens and ngrams etc.

    Returns:
        dict: Tweet with formatted fields
    """

    subj_sent_check = args[0]
    result = tweet_obj

    if subj_sent_check:
        subjectivity = args[1]
        sentiment = args[2]
        result["subjectivity"] = subjectivity
        result["compound_score"] = sentiment["compound"]
        result["neg_score"] = sentiment["neg"]
        result["neu_score"] = sentiment["neu"]
        result["pos_score"] = sentiment["pos"]

    result["hs_keyword_count"] = len(processed_text[4])
    result["hs_keyword_matches"] = processed_text[4]
    result["tokens"] = processed_text[0]
    result["stopwords"] = processed_text[1]
    result["hashtags"] = processed_text[2]
    result["user_mentions"] = processed_text[3]
    result["unigrams"] = processed_text[5][0]
    result["bigrams"] = processed_text[5][1]
    result["trigrams"] = processed_text[5][2]
    result["quadgrams"] = processed_text[5][3]
    result["pentagrams"] = processed_text[5][4]
    result["stopword_ngrams"] = processed_text[6]

    return result


def create_ngrams(text_list, length):
    """ Create ngrams of the specified length from a string of text
    Args:
        text_list   (list): Pre-tokenized text to process.
        length      (int):  Length of ngrams to create.
    """

    clean_tokens = [token for token in text_list if token not in PUNCTUATION]
    return [" ".join(i for i in ngram) for ngram in ngrams(clean_tokens, length)]


def do_create_ngram_collections(text, args):
    """ Create and return ngram collections and set intersections.
    Text must be lowercased if required.
    Args:
        text   (str): Text to process.
        args      (list): Can contains the following:
            0: porn_black_list (list): List of porn keywords.
            1: hs_keywords (list) HS corpus.
            2: black_list  (list) Custom words to filter on.
    """

    porn_black_list = args[0]
    hs_keywords = args[1]
    if args[2]:
        black_list = args[2]

    tokens = twokenize.tokenizeRawTweetText(text)
    unigrams = create_ngrams(tokens, 1)
    # bigrams = create_ngrams(tokens, 2)
    trigrams = create_ngrams(tokens, 3)
    quadgrams = create_ngrams(tokens, 4)

    xgrams = trigrams + quadgrams
    unigrams = set(unigrams)
    xgrams = set(xgrams)

    # Set operations are faster than list iterations.
    # Here we perform a best effort series of filters
    # to ensure we only get tweets we want.
    # unigram_intersect = set(porn_black_list).intersection(unigrams)
    xgrams_intersect = porn_black_list.intersection(xgrams)
    hs_keywords_intersect = hs_keywords.intersection(unigrams)

    if args[2]:
        black_list_intersect = black_list.intersection(unigrams)
    else:
        black_list_intersect = None

    return [None, xgrams_intersect, hs_keywords_intersect, black_list_intersect]


def is_garbage(raw_text, precision):
    """ Check if a tweet consists primarly of hashtags, mentions or urls

    Args:
        tweet_obj  (dict): Tweet to preprocess.
    """

    word_list = raw_text.split()
    garbage_check = [
        token for token in word_list if not token.startswith(("#", "@", "http"))]
    garbage_check = " ".join(garbage_check)

    if ((len(raw_text) - len(garbage_check)) / len(raw_text)) >= precision:
        return True
    else:
        return False


def clean_tweet_text(raw_text):
    """Clean up tweet text and preserve emojis"""

    # Remove urls
    clean_text = remove_urls(raw_text)
    clean_text = twokenize.tokenizeRawTweetText(clean_text)

    # Remove numbers
    clean_text = [token for token in clean_text if len(
        token.strip(string.digits)) == len(token)]

    # Record single instances of a term only
    terms_single = set(clean_text)
    terms_single = dict.fromkeys(terms_single)

    # Store any emoji in the text and prevent it from being lowercased then
    # append items marked as Protected by the twokenize library.
    # This protected object covers tokens that shouldn't be split or lowercased so
    # we take advantage of it and use it as quick and dirty way to
    # identify emoji rather than calling a separate regex function.
    emoji = []
    for token in clean_text:
        for _match in twokenize.Protected.finditer(token):
            emoji.append(token)

    # If the token is not in the emoji dict, lowercase it
    # TODO preserve tokens with all uppercase chars as a measure of intensity
    emoji = dict.fromkeys(emoji)
    for token in terms_single.copy():
        if not token in emoji:
            terms_single[token.lower()] = terms_single.pop(token)

    return [clean_text, terms_single, emoji]


def remove_urls(raw_text):
    """ Removes urls from text

    Args:
        raw_text (str): Text to filter.
    """
    return re.sub(
        r"(?:http|https):\/\/((?:[\w-]+)(?:\.[\w-]+)+)(?:[\w.,@?^=%&amp;:\/~+#-]*[\w@?^=%&amp;\/~+#-])?", "", raw_text)


def get_emotion_coverage(tweet_obj, projection):
    """Send the text through an emotion API and return the results"""

    emotion_detector = EmotionDetection.EmotionDetection()
    result = emotion_detector.get_emotion_json(tweet_obj[projection])

    if result["ambiguous"] == "no":

        if len(result["groups"]) == 2:
            return UpdateOne({"_id": tweet_obj["_id"]}, {
                "$set": {"primary_emo_group": result["groups"][0]["name"],
                         "primary_emos": result["groups"][0]["emotions"],
                         "secondary_emo_group": result["groups"][1]["name"],
                         "secondary_emos":  result["groups"][1]["emotions"]
                         }}, upsert=False)

        elif len(result["groups"]) == 1:
            return UpdateOne({"_id": tweet_obj["_id"]}, {
                "$set": {"primary_emo_group": result["groups"][0]["name"],
                         "primary_emos": result["groups"][0]["emotions"]
                         }}, upsert=False)
    else:
        return UpdateOne({"_id": tweet_obj["_id"]}, {
            "$set": {"emotion_ambiguous": True}}, upsert=False)


def parallel_preprocess(tweet_list, hs_keywords, subj_check, sent_check):
    """Passes the incoming raw tweets to our preprocessing function.

    Args:
        tweet_list  (list): List of raw tweet texts to preprocess.
        tweet_split (list): List of tokens in the tweet text.
        hs_keywords (set): Keywords to match on.
        subj_check (bool): Check text for subjectivity.
        sent_check (bool): Check text for sentiment.

    Returns:
        list: List of vectorized tweets.
    """
    # num_cores = cpu_count()
    results = Parallel(n_jobs=1)(
        delayed(preprocess_tweet)(tweet, hs_keywords, [subj_check, sent_check]) for tweet in tweet_list)
    return results


def parallel_emotion_coverage(tweet_list, projection):
    """Passes the incoming raw tweets to our preprocessing function.

    Args:
        tweet_list  (list): List of raw tweet texts to preprocess.
        projection (string): Tweet object property to access.

    Returns:
        results: List of emotion tagged tweets.
    """
    num_cores = cpu_count()
    results = Parallel(n_jobs=num_cores, backend="threading")(
        delayed(get_emotion_coverage)(tweet, projection) for tweet in tweet_list)
    return results


def get_similar_words(word, k_words):
    """Return the top k similar words as stored in the spaCy gloVe vectors
    Args:
        doc (spaCy Doc): A container for accessing linguistic annotations.
        k_words (int): Number of words to return.
    https://github.com/explosion/spaCy/issues/276
    """
    queries = [w for w in word.vocab if w.has_vector and w.lower_ !=
               word.lower_ and w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = sorted(
        queries, key=lambda w: word.similarity(w), reverse=True)
    return by_similarity[:k_words]


def get_keywords(doc):
    """ Returns the keywords in a document by exploiting doc.noun_chunks
    Args:
        doc (spaCy Doc): A container for accessing linguistic annotations.

    Returns:
        result: set of stopword filtered keywords.
    """
    result = set()
    # Here we check for tokens that are spans, ie. more than one word
    # separated by " "
    for span in doc.noun_chunks:
        if " " in span.text:
            split = span.text.split(" ")
            for token in split:
                result.add(token)
        else:
            # This returns the token at the specified indexed
            get_doc_token = doc[token.start]
            if not (get_doc_token.is_oov or get_doc_token.like_num or get_doc_token.lower not in STOP_LIST):
                result.add(get_doc_token.lower_)
    return result


def count_uppercase_tokens(doc):
    """Count the number of tokens with all uppercase
     Args:
        doc (spaCy Doc): A container for accessing linguistic annotations.
    Returns:
        sum: count of uppercase tokens
    """
    count = 0
    for token in doc:
        if token.text.isupper() and len(token) != 1:
            count += 1
    return count
