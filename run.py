import os
import json
import time
import glob
import csv
import itertools
import cProfile
from pprint import pprint
import pymongo
from bson.son import SON
from bson.code import Code

JSON_PATH = "HatebaseData/json/"
CSV_PATH = "HatebaseData/csv/"
DATA_PATH = "Data/"
DB_URL = "mongodb://140.114.79.146:27017"
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 50
USER_MENTIONS_LIMIT = 50


def timing(func):
    """Decorator for timing run time of a function
    """
    def wrap(*args):
        """Wrapper
        """
        time1 = time.time()
        ret = func(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (func.func_name, (time2 - time1) * 1000.0)
        return ret
    return wrap


def do_cprofile(func):
    """Decorator for profiling a function
    """
    def profiled_func(*args, **kwargs):
        """Wrapper
        """
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='time')
    return profiled_func


def connect():
    """Initializes a pymongo conection object.

    Returns:
        pymongo.MongoClient: Connection object for Mongo DB_URL
    """
    try:
        conn = pymongo.MongoClient(DB_URL)
        print "Connected to DB at " + DB_URL + " successfully"
    except pymongo.errors.ConnectionFailure, ex:
        print "Could not connect to MongoDB: %s" % ex
    return conn


def get_filenames():
    """Reads all the json files in the folder and removes the drive and path and
    extension, only returning a list of strings with the file names.

    Returns:
        list: List of plain filenames
    """
    file_path = glob.glob(JSON_PATH + "*.json")
    result = []
    for entry in file_path:
        drive, path = os.path.splitdrive(entry)
        path, filename = os.path.split(path)
        name = os.path.splitext(filename)[0]
        result.append(str(name))
    return result


def read_json_file(filename, path):
    """Accepts a file name and loads it as a json object.
    Args:
        filename   (str): Filename to be loaded.
        path       (str): Directory path to use.

    Returns:
        list: List of words concatenated with OR.
    """

    result = []
    try:
        with open(path + filename + '.json', 'r') as entry:
            result = json.load(entry)
    except IOError as ex:
        print "I/O error({0}): {1}".format(ex.errno, ex.strerror)
    else:
        entry.close()
        return result


def write_json_file(filename, path, result):
    """Writes the result to json with the given filename.
    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """

    with open(path + filename + '.json', 'w+') as json_file:
        json.dump(result, json_file)
    json_file.close()


def read_csv_file(filename, path):
    """Accepts a file name and loads it as a list.
    Args:
        filename   (str): Filename to be loaded.
        path       (str): Directory path to use.

    Returns:
        list: List of strings.
    """

    try:
        with open(path + filename + '.csv', 'r') as entry:
            reader = csv.reader(entry)
            temp = list(reader)
            # flatten to 1D, it gets loaded as 2D array
            result = [x for sublist in temp for x in sublist]
    except IOError as ex:
        print "I/O error({0}): {1}".format(ex.errno, ex.strerror)
    else:
        entry.close()
        return result


def write_csv_file(filename, result, path):
    """Writes the result to csv with the given filename.
    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """

    output = open(path + filename + '.csv', 'wb')
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator='\n')
    for val in result:
        writer.writerow([val])
    # Print one a single row
    # writer.writerow(result)


def extract_corpus(file_list):
    """ Loads a set of json files and builds a corpus from the terms within
    Args:
        file_list  (list): list of file names.
    """

    for entry in file_list:
        json_data = read_json_file(entry, JSON_PATH)
        result = []

        data = json_data['data']['datapoint']
        data.sort(key=count_sightings, reverse=True)

        for entry in data:
            result.append(str(entry['vocabulary']))
        write_csv_file(entry, result, CSV_PATH)


def count_sightings(json_obj):
    """ Returns a count of the number of sightings per word in corpus
    Args:
        json_obj (json_obj).

    Returns:
        int: The return value.
    """
    try:
        return int(json_obj['number_of_sightings'])
    except KeyError:
        return 0


def count_entries(file_list):
    """Performs a count of the number of number of words in the corpus
     Args:
        file_list  (list): list of file names.

    Returns:
        list: A list of json objects containing the count per file name
    """
    result = []
    for obj in file_list:
        with open(CSV_PATH + obj + '.csv', "r") as entry:
            reader = csv.reader(entry, delimiter=",")
            col_count = len(reader.next())
            res = {"Filename": obj, "Count": col_count}
            result.append(res)
    return result


def build_query_string(query_words):
    """Builds an OR concatenated string for querying the Twitter Search API.
    Args:
        query_words (list): list of words to be concatenated.

    Returns:
        list: List of words concatenated with OR.
    """
    result = ''.join(
        [q + ' OR ' for q in query_words[0:(len(query_words) - 1)]])
    return result + str(query_words[len(query_words) - 1])


def test_file_operations():
    """
    Test previous methods
    """
    file_list = get_filenames()
    extract_corpus(file_list)
    num_entries = count_entries(file_list)
    pprint(num_entries)
    res = read_csv_file('about_sexual_orientation_eng_pg1', CSV_PATH)
    res2 = build_query_string(res)
    print res2


def unicode_to_utf(unicode_list):
    """ Converts a list of strings from unicode to utf8
    Args:
        unicode_list (list): A list of unicode strings.

    Returns:
        list: UTF8 converted list of strings.
    """
    return [x.encode('UTF8') for x in unicode_list]


def get_language_list(client, db_name):
    """Returns a list of all the matching languages within the collection
     Args:
        client  (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name (str): Name of database to query.

    Returns:
        list: List of languages within the twitter collection.
    """
    dbo = client[db_name]
    distinct_lang = dbo.tweets.distinct("lang")
    return unicode_to_utf(distinct_lang)


def get_language_distribution(client, db_name, lang_list):
    """Returns the distribution of tweets matching either
    english, undefined or spanish.

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.

    Returns:
        list: Distribution for each language in lang_list.
    """

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$group": {"_id": "$lang", "count": {"$sum": 1}}},
        {"$project": {"language": "$_id", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("language", -1)])}
    ]
    return dbo.tweets.aggregate(pipeline)


def test_get_language_distribution(client):
    """Test and print results of aggregation

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = get_language_list(client, 'twitter')
    cursor = get_language_distribution(client, 'twitter', lang_list)
    write_json_file('language_distribution', DATA_PATH, list(cursor))


def test_get_language_subset(client):
    """Test and print the results of aggregation
    Constrains language list to en, und, es.

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = ['en', 'und', 'es']
    cursor = get_language_distribution(client, 'twitter', lang_list)
    for document in cursor:
        print document


def create_lang_subset(client, db_name, lang):
    """Subsets the collection by the specified language.
    Outputs value to new collection

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang        (list): language to match on.

    """

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": lang}},
        {"$out": "subset_" + lang}
    ]
    dbo.tweets.aggregate(pipeline)


def get_top_k_users(client, db_name, lang_list, k_filter, limit):
    """Finds the top k users in the collection.
    k_filter is the name of an array in the collection, we apply the $unwind operator to it

     Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.
        k_filter    (str):  Name of an array in the collection.abs
        limit       (int):  Limit for the number of results to return.

    Returns:
        list: List of objects containing id_str, screen_name and the frequency of appearance.
    """
    k_filter_base = k_filter
    k_filter = "$" + k_filter
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {k_filter_base: 1, "_id": 0}},
        {"$unwind": k_filter},
        {"$group": {"_id": {"id_str": k_filter + ".id_str", "screen_name":
                            k_filter + ".screen_name"}, "count": {"$sum": 1}}},
        {"$project": {"id_str": "$_id.id_str",
                      "screen_name": "$_id.screen_name", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("id_str", -1)])}
    ]
    return dbo.tweets.aggregate(pipeline, allowDiskUse=True)


def get_top_k_hashtags(client, db_name, lang_list, k_filter, limit, k_value):
    """Finds the top k hashtags in the collection.
    k_filter is the name of an array in the collection, we apply the $unwind operator to it

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.
        k_filter    (str):  Name of an array in the collection.abs
        limit       (int):  Limit for the number of results to return.
        k_value     (int):  Filter for the number of occurences for each hashtag

    Returns:
        list: List of objects containing _id, hashtag text and the frequency of appearance.
    """

    k_filter_base = k_filter
    k_filter = "$" + k_filter
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {k_filter_base: 1, "_id": 0}},
        {"$unwind": k_filter},
        {"$group": {"_id": k_filter + ".text", "count": {"$sum": 1}}},
        {"$project": {"hashtag": "$_id", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("_id", -1)])},
        {"$match": {"count": {"$gt": k_value}}},
        {"$limit": limit},
    ]
    return dbo.tweets.aggregate(pipeline)


def test_get_top_k_users(client, db_name, lang_list, k_filter):
    """Test and print results of top k aggregation
    """
    cursor = get_top_k_users(client, db_name, lang_list,
                             k_filter, USER_MENTIONS_LIMIT)
    write_json_file('user_distribution', DATA_PATH, list(cursor))


@do_cprofile
def test_get_top_k_hashtags(client, db_name, lang_list, k_filter, k_value):
    """Test and print results of top k aggregation
    """
    cursor = get_top_k_hashtags(
        client, db_name, lang_list, k_filter, HASHTAG_LIMIT, k_value)
    write_json_file('hashtag_distribution', DATA_PATH, list(cursor))


def user_mentions_map_reduce(client, db_name, subset, output_name):
    """Map reduce that returns the number of times a user is mentioned

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.

    Returns:
        list: List of objects containing _id and the frequency of appearance.
    """
    map_function = Code("function () {"
                        "    var userMentions = this.entities.user_mentions;"
                        "    for (var i = 0; i < userMentions.length; i ++){"
                        "        if (userMentions[i].screen_name.length > 0) {"
                        "            emit (userMentions[i].screen_name, 1);"
                        "        }"
                        "    }"
                        "}")

    reduce_function = Code("function (keyUsername, occurs) {"
                           "     return Array.sum(occurs);"
                           "}")
    frequency = []
    dbo = client[db_name]
    cursor = dbo[subset].map_reduce(
        map_function, reduce_function, output_name)

    for document in cursor.find():
        frequency.append({'_id': document['_id'], 'value': document['value']})

    frequency = sorted(frequency, key=lambda k: k['value'], reverse=True)
    write_json_file('user_distribution_mr', DATA_PATH, frequency)
    pprint(frequency)


def hashtag_map_reduce(client, db_name, subset, output_name):
    """Map reduce that returns the number of times a hashtag is used

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.

    Returns:
        list: List of objects containing _id and the frequency of appearance.
    """
    map_function = Code("function () {"
                        "    var hashtags = this.entities.hashtags;"
                        "    for (var i = 0; i < hashtags.length; i ++){"
                        "        if (hashtags[i].text.length > 0) {"
                        "            emit (hashtags[i].text, 1);"
                        "        }"
                        "    }"
                        "}")

    reduce_function = Code("function (keyHashtag, occurs) {"
                           "     return Array.sum(occurs);"
                           "}")
    frequency = []
    dbo = client[db_name]
    cursor = dbo[subset].map_reduce(
        map_function, reduce_function, output_name)

    for document in cursor.find():
        frequency.append({'_id': document['_id'], 'value': document['value']})

    frequency = sorted(frequency, key=lambda k: k['value'], reverse=True)
    write_json_file('hashtag_distribution_mr', DATA_PATH, frequency)
    pprint(frequency)


def collection_finder(client, db_name, subset):
    """Fetches the specified collection
    """
    dbo = client[db_name]
    # cursor = dbo[subset].find({"count":{"$gt":500}})
    # cursor = dbo[subset].find(
    #     {}, no_cursor_timeout=True)
    # cursor.batch_size(80000)
    write_json_file(subset, DATA_PATH, list(
        dbo[subset].find({}, {"hashtag": 1, "count": 1, "_id": 0})))


def main():
    """
    Test functionality
    """
    client = connect()
    # test_get_language_distribution(client)
    # test_get_language_subset(client)
    # create_lang_subset(client, 'twitter', 'ru')
    # user_mentions_map_reduce(client, 'twitter', 'subset_ru')
    # hashtag_map_reduce(client, 'twitter', 'subset_ru', 'hashtag_ru')
    # test_get_top_k_users(client, 'twitter', ['ru'], USER_MENTIONS)
    # test_get_top_k_hashtags(client, 'twitter', ['ru'], HASHTAGS, 20)
    collection_finder(client, 'twitter', 'hashtag_dist_en')

if __name__ == '__main__':
    main()
