import os
import json
import glob
import csv
from pprint import pprint
import collections
import pymongo
from bson.son import SON
from bson.code import Code

JSON_PATH = "HatebaseData/json/"
CSV_PATH = "HatebaseData/csv/"
DATA_PATH = "Data/"
DB_URL = ***REMOVED***
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 10000
USER_MENTIONS_LIMIT = 10000


def connect():
    """Return connection object for Mongo DB"""
    try:
        conn = pymongo.MongoClient(DB_URL)
        print "Connected to DB at " + DB_URL + " successfully"
    except pymongo.errors.ConnectionFailure, ex:
        print "Could not connect to MongoDB: %s" % ex
    return conn


def get_filenames():
    """
    Reads all the json files in the folder and removes the drive and path and
    extension, only returning a list of strings with the file names.
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
    """
    Accepts a file name and loads it as a json object
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


def write_json_file(filename, result, path):
    """
    Writes the result to json with the given filename
    """
    with open(path + filename + '.json', 'w+') as json_file:
        json.dump(result, json_file)
    json_file.close()


def read_csv_file(filename, path):
    """
    Accepts a file name and loads it as a list
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
    """
    Writes a list to csv with the given filename
    """
    output = open(path + filename + '.csv', 'wb')
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator='\n')
    for val in result:
        writer.writerow([val])
    # Print one a single row
    # writer.writerow(result)


def extract_corpus(file_list):
    """
    Loads a set of json files and builds a corpus from the
    terms within
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
    """
    Returns a count of the number of sightings per word in corpus
    """
    try:
        return int(json_obj['number_of_sightings'])
    except KeyError:
        return 0


def count_entries(file_list):
    """
    Performs a count of the number of number of words in the corpus
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
    """
    Builds an OR concatenated string for querying the Twitter Search API
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
    """
    Converts a list of strings from unicode to utf8
    """
    return [x.encode('UTF8') for x in unicode_list]


def get_language_list(client, db_name):
    """
    Returns a list of all the matching languages within the collection
    """
    dbo = client[db_name]
    distinct_lang = dbo.tweets.distinct("lang")
    return unicode_to_utf(distinct_lang)


def get_language_distribution(client, db_name, lang_list):
    """
    Returns the distribution of tweets matching either
    english, undefined or spanish
    """
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$group": {"_id": "$lang", "count": {"$sum": 1}}},
        {"$project": {"language": "$_id", "count": 1, "_id": 0}}
    ]
    return dbo.tweets.aggregate(pipeline)


def test_get_language_distribution(client):
    """
    Test and print results of aggregation
    """
    lang_list = get_language_list(client, 'twitter')
    cursor = get_language_distribution(client, 'twitter', lang_list)
    for document in cursor:
        print document


def test_get_language_subset(client):
    """
    Test and print the results of aggregation
    Constrains language list to en, und, es
    """
    lang_list = ['en', 'und', 'es']
    cursor = get_language_distribution(client, 'twitter', lang_list)
    for document in cursor:
        print document


def create_lang_subset(client, db_name, lang):
    """
    Subsets the collection by the specified language
    Outputs value to new collection
    """
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": lang}},
        {"$out": "subset_" + lang}
    ]
    dbo.tweets.aggregate(pipeline)


def get_top_k_users(client, db_name, lang_list, k_filter, limit):
    """
    Finds the top k users in the collection
    k_filter is the name of an array in the collection, we apply the $unwind operator to it
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
        {"$sort": SON([("count", -1), ("_id", -1)])},
        {"$limit": limit},
    ]
    return dbo.tweets.aggregate(pipeline, allowDiskUse=True)


def get_top_k_hashtags(client, db_name, lang_list, k_filter, limit):
    """
    Finds the top k hashtags in the collection
    k_filter is the name of an array in the collection, we apply the $unwind operator to it
    """
    k_filter_base = k_filter
    k_filter = "$" + k_filter
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {k_filter_base: 1, "_id": 0}},
        {"$unwind": k_filter},
        {"$group": {"_id": k_filter + ".text", "count": {"$sum": 1}}},
        {"$sort": SON([("num", -1), ("_id", -1)])},
        {"$limit": limit},
    ]
    return dbo.tweets.aggregate(pipeline, allowDiskUse=True)


def test_get_top_k_users(client, db_name, lang_list, k_filter):
    """
    Test and print results of top k aggregation
    """
    frequency = []
    # names = []
    cursor = get_top_k_users(client, db_name, lang_list,
                             k_filter, USER_MENTIONS_LIMIT)
    for document in cursor:
        frequency.append({'screen_name': document['_id']['screen_name'],
                          'value': document['count'], '_id': document['_id']['id_str']})
        # names.append(document['_id']['screen_name'])
    pprint(frequency)
    write_json_file('user_distribution', frequency, DATA_PATH)
    # Check for duplicates
    # print [item for item, count in collections.Counter(names).items() if
    # count > 1]


def test_get_top_k_hashtags(client, db_name, lang_list, k_filter):
    """
    Test and print results of top k aggregation
    """
    frequency = []
    cursor = get_top_k_hashtags(
        client, db_name, lang_list, k_filter, HASHTAG_LIMIT)
    for document in cursor:
        frequency.append({'hashtag': document['_id'],
                          'value': document['count']})
    pprint(frequency)
    write_json_file('hashtag_distribution', frequency, DATA_PATH)


def sample_map_reduce(client, db_name, subset):
    """
    Map reduce that returns the number of times a user is mentioned
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
        map_function, reduce_function, "out:{ inline: 1 }")
    for document in cursor.find():
        frequency.append({'_id': document['_id'], 'value': document['value']})
    pprint(frequency)


def main():
    """
    Test functionality
    """
    client = connect()
    # sample_map_reduce(client, 'twitter', 'subset_gu')
    # test_get_language_distribution(client)
    # test_get_language_subset(client)
    # create_lang_subset(client, 'twitter', 'gu')
    test_get_top_k_users(client, 'twitter', ['gu'], USER_MENTIONS)
    test_get_top_k_hashtags(client, 'twitter', ['gu'], HASHTAGS)

if __name__ == '__main__':
    main()
