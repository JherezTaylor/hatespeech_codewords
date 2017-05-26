# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses various io functions for use throughout the project
"""

import os
import csv
from collections import OrderedDict
import glob
import ujson
import pandas as pd
from ..db import mongo_base
from . import settings
from . import notifiers


def unicode_to_utf(unicode_list):
    """ Converts a list of strings from unicode to utf8

    Args:
        unicode_list (list): A list of unicode strings.

    Returns:
        list: UTF8 converted list of strings.
    """
    return [x.encode("UTF8") for x in unicode_list]


def count_sightings(json_obj):
    """ Returns a count of the number of sightings per word in corpus

    Args:
        json_obj (dict).

    Returns:
        int: The return value.
    """
    try:
        return int(json_obj["number_of_sightings"])
    except KeyError:
        return 0


def build_query_string(query_words):
    """Builds an OR concatenated string for querying the Twitter Search API.

    Args:
        query_words (list): list of words to be concatenated.

    Returns:
        list: List of words concatenated with OR.
    """
    result = "".join(
        [q + " OR " for q in query_words[0:(len(query_words) - 1)]])
    return result + str(query_words[len(query_words) - 1])


def prep_json_entry(entry):
    """Properly format and return a json object
    """

    json_obj = OrderedDict()
    json_obj["vocabulary"] = entry["vocabulary"]
    json_obj["variant_of"] = entry["variant_of"]
    json_obj["pronunciation"] = entry["pronunciation"]
    json_obj["meaning"] = entry["meaning"]
    json_obj["language"] = entry["language"]
    json_obj["about_ethnicity"] = entry["about_ethnicity"]
    json_obj["about_nationality"] = entry["about_nationality"]
    json_obj["about_religion"] = entry["about_religion"]
    return json_obj


@notifiers.do_cprofile
def get_filenames(directory):
    """Reads all the json files names in the directory.

    Returns:
        list: List of plain filenames
    """
    file_path = glob.glob(directory + "*.json")
    result = []

    for entry in file_path:
        _, path = os.path.splitdrive(entry)
        path, filename = os.path.split(path)
        name = os.path.splitext(filename)[0]
        result.append(str(name))
    return result


@notifiers.do_cprofile
def read_json_file(filename, path):
    """Accepts a file name and loads it as a json object.

    Args:
        filename   (str): Filename to be loaded.
        path       (str): Directory path to use.

    Returns:
        obj: json object
    """

    result = []
    try:
        with open(path + filename + ".json", "r") as entry:
            result = ujson.load(entry)
    except IOError as ex:
        print("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    else:
        entry.close()
        return result


def write_json_file(filename, path, result):
    """Writes the result to json with the given filename.

    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """

    with open(path + filename + ".json", "w+") as json_file:
        ujson.dump(result, json_file)
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
        with open(path + filename + ".csv", "r") as entry:
            reader = csv.reader(entry)
            temp = list(reader)
            # flatten to 1D, it gets loaded as 2D array
            result = [x for sublist in temp for x in sublist]
    except IOError as ex:
        print("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    else:
        entry.close()
        return result


def write_csv_file(filename, path, result):
    """Writes the result to csv with the given filename.

    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """

    output = open(path + filename + ".csv", "w")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\n")
    for val in result:
        writer.writerow([val])
    # Print one a single row
    # writer.writerow(result)


def count_entries(file_list):
    """Performs a count of the number of number of words in the corpus.

    Args:
        file_list  (list): list of file names.

    Returns:
        list: A list of json objects containing the count per file name
    """

    result = []
    for obj in file_list:
        with open(settings.CSV_PATH + obj + ".csv", "r") as entry:
            reader = csv.reader(entry, delimiter=",")
            col_count = len(next(reader))
            res = {"Filename": obj, "Count": col_count}
            result.append(res)
    return result


def extract_corpus(file_list):
    """ Loads a set of json files and builds a corpus from the terms within.

    Args:
        file_list  (list): list of file names.
    """

    for file_name in file_list:
        json_data = read_json_file(file_name, settings.JSON_PATH)
        result = []

        data = json_data["data"]["datapoint"]
        data.sort(key=count_sightings, reverse=True)

        result = [str(entry["vocabulary"]) for entry in data]
        write_csv_file(file_name, settings.CSV_PATH, result)


def json_field_filter(json_obj, field_filter):
    """Accepts a json object and returns only the passed field

    Args:
        json_obj        (obj): json object.
        field_filter    (str): field to extract.

    Returns:
        list: A list of filtered values
    """
    result = [document[field_filter] for document in json_obj]
    return result


def filter_hatebase_categories():
    """Filters the hatebase data into categories.

    Does manually parsing for black, muslim and latino keywords.
    """
    filter1_subset = []
    filter2_subset = []
    filter3_subset = []

    seen_set = set()

    filter1 = ["Black", "black", "Blacks", "blacks",
               "African", "african", "Africans", "africans"]
    filter2 = ["Muslims", "Muslim", "Middle", "Arab", "Arabs", "Arabic"]
    filter3 = ["Hispanic", "hispanic", "Hispanics", "Mexican", "Mexicans", "Latino", "Latinos",
               "Cuban", "Cubans"]

    file_list = get_filenames(settings.JSON_PATH)
    for entry in file_list:
        json_data = read_json_file(entry, settings.JSON_PATH)

        data = json_data["data"]["datapoint"]
        data.sort(key=count_sightings, reverse=True)

        for entry in data:
            if any(x in entry["meaning"] for x in filter1):
                if entry["vocabulary"] not in seen_set:
                    seen_set.add(entry["vocabulary"])
                    filter1_subset.append(prep_json_entry(entry))

            if any(x in entry["meaning"] for x in filter2):
                if entry["vocabulary"] not in seen_set:
                    seen_set.add(entry["vocabulary"])
                    filter2_subset.append(prep_json_entry(entry))

            if any(x in entry["meaning"] for x in filter3):
                if entry["vocabulary"] not in seen_set:
                    seen_set.add(entry["vocabulary"])
                    filter3_subset.append(prep_json_entry(entry))

    write_json_file(
        "filter1_subset", settings.OUTPUT_PATH, filter1_subset)
    write_json_file(
        "filter2_subset", settings.OUTPUT_PATH, filter2_subset)
    write_json_file(
        "filter3_subset", settings.OUTPUT_PATH, filter3_subset)


def parse_category_files():
    """Reads the category entries and return the keywords only

    Returns:
        list: A list of filtered keywords
    """
    result = []
    filter1 = json_field_filter(read_json_file(
        "filter1_subset", settings.OUTPUT_PATH), "vocabulary")
    filter2 = json_field_filter(read_json_file(
        "filter2_subset", settings.OUTPUT_PATH), "vocabulary")
    filter3 = json_field_filter(read_json_file(
        "filter3_subset", settings.OUTPUT_PATH), "vocabulary")

    result = filter1 + filter2 + filter3
    return result


def fetch_as_df(connection_params, projection):
    """ Takes MongoDB connection params and returns the specified
    collection as a pandas dataframe.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        projection (dict): Dictionary of fields to return, returns all fields if blank.
    """

    client = mongo_base.connect()
    connection_params.insert(0, client)
    query = {}
    query["filter"] = {}
    query["projection"] = projection
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True
    cursor = mongo_base.finder(connection_params, query, False)
    _df = pd.DataFrame(list(cursor))
    return _df
