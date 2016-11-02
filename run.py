import json, os, glob, pprint

def get_file_names():
    """
    Reads all the json files in the folder and removes the drive and path and
    extension, only returning a list of strings with the file names.
    """
    read_files = glob.glob("*.json")
    result = []
    for m in read_files:
        drive, path = os.path.splitdrive(m)
        path, filename = os.path.split(path)
        name = os.path.splitext(filename)[0]
        result.append(str(name))
    return result


def load_json_file(file_name):
    data = []
    try:
        with open(file_name+'.json', 'r') as f:
            data = json.load(f)
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    else:
        f.closed
        return data

# item_dict = json.loads(json_data)
# print len(item_dict['result'][0]['run'])


def main():
    # file_list = get_file_names()
    # print file_list
    hlist = load_json_file('about_class_eng-pg1')
    pprint.pprint(hlist)

if __name__ == '__main__':
   main()
