import json
import os
import glob
import pprint
import csv
import pymongo

JSON_PATH = "HatebaseData/json/"
CSV_PATH = "HatebaseData/csv/"
DB_URL = "mongodb://140.114.79.146:27017"


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


def load_json_file(filename):
    """
    Accepts a file name and loads it as a json object
    """
    result = []
    try:
        with open(JSON_PATH + filename + '.json', 'r') as entry:
            result = json.load(entry)
    except IOError as ex:
        print "I/O error({0}): {1}".format(ex.errno, ex.strerror)
    else:
        entry.close()
        return result


def write_to_csv(filename, result):
    """
    Writes a list to csv with the given filename
    """
    output = open(CSV_PATH + filename + '.csv', 'wb')
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
        json_data = load_json_file(entry)
        result = []

        data = json_data['data']['datapoint']
        data.sort(key=count_sightings, reverse=True)

        for entry in data:
            result.append(str(entry['vocabulary']))
        write_to_csv(entry, result)


def count_sightings(json_obj):
    """
    Returns a count of the number of sightings per word in corpus
    """
    try:
        return int(json_obj['number_of_sightings'])
    except KeyError:
        return 0


def load_csv_file(filename):
    """
    Accepts a file name and loads it as a list
    """
    try:
        with open(CSV_PATH + filename + '.csv', 'r') as entry:
            reader = csv.reader(entry)
            temp = list(reader)
            # flatten to 1D, it gets loaded as 2D array
            result = [x for sublist in temp for x in sublist]
    except IOError as ex:
        print "I/O error({0}): {1}".format(ex.errno, ex.strerror)
    else:
        entry.close()
        return result


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
    pprint.pprint(num_entries)
    res = load_csv_file('about_sexual_orientation_eng_pg1')
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


def main():
    """
    Test functionality
    """
    client = connect()
    test_get_language_subset(client)

if __name__ == '__main__':
    main()
