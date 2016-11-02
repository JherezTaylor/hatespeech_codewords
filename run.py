import json
import os
import glob
import pprint
import csv

json_path = "HatebaseData/json/"
csv_path = "HatebaseData/csv/"


def get_filenames():
    """
    Reads all the json files in the folder and removes the drive and path and
    extension, only returning a list of strings with the file names.
    """
    file_path = glob.glob(json_path + "*.json")
    result = []
    for f in file_path:
        drive, path = os.path.splitdrive(f)
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
        with open(json_path + filename + '.json', 'r') as f:
            result = json.load(f)
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    else:
        f.closed
        return result


def write_to_csv(filename, result):
    """
    Writes a list to csv with the given filename
    """
    output = open(csv_path + filename + '.csv', 'wb')
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(result)


def extract_corpus(file_list):
    """
    Loads a set of json files and builds a corpus from the
    terms within
    """
    for f in file_list:
        json_data = load_json_file(f)
        result = []
        for entry in json_data['data']['datapoint']:
            result.append(entry['vocabulary'])
        write_to_csv(f, result)


def main():
    file_list = get_filenames()
    extract_corpus(file_list)

if __name__ == '__main__':
    main()
