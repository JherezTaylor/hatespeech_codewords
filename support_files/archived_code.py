def keyword_search(connection_params, keyword_list, lang_list):
    """Perform a text search with the provided keywords.

    We also preprocess the tweet text in order to avoid redundant operations.
    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        keyword_list    (list): List of keywords to search for.
        lang_list       (list): List of languages to match on.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    # Store the documents for our bulkwrite
    operations = []
    # Keep track of the tweets that we have already seen, keep distinct.
    seen_set = set()
    dbo = client[db_name]
    for search_query in keyword_list:
        # Run an aggregate search for each keyword, might get better performance
        # from running n keywords at a time, but I'm not sure.
        pipeline = [
            {"$match": {"$and": [{"$text": {"$search": search_query}},
                                 {"id_str": {"$nin": list(seen_set)}},
                                 {"lang": {"$in": lang_list}},
                                 {"retweet_count": 0}]}},
            {"$project": {"_id": 1, "id_str": 1, "text": 1, "id": 1, "timestamp": 1,
                          "lang": 1, "user.id_str": 1, "user.screen_name": 1, "user.location": 1}},
            {"$out": "temp_set"}
        ]
        dbo[collection].aggregate(pipeline, allowDiskUse=True)

        cursor = dbo["temp_set"].find({}, no_cursor_timeout=True)
        entities = cursor[:]

        print("Keyword:", search_query, "| Count:", cursor.count(), " | Seen:", len(seen_set))
        for document in entities:
            seen_set.add(document["id_str"])
            # Create a new field and add the preprocessed text to it
            operations.append(document)

            # document["vector"] = text_preprocessing.preprocess_text(document["text"])
            # operations.append(InsertOne(document))

            # Send once every 1000 in batch
            if (len(operations) % 1000) == 0:
                operations = text_preprocessing.parallel_preprocess(operations)
                dbo["keyword_collection"].bulk_write(operations, ordered=False)
                operations = []

    if (len(operations) % 1000) != 0:
        operations = text_preprocessing.parallel_preprocess(operations)
        dbo["keywords_collection"].bulk_write(operations, ordered=False)

    # Clean Up
    dbo["temp_set"].drop()


def filter_by_language(connection_params, lang_list, output_name):
    """Aggregation pipeline to remove tweets with a lang field not in
    lang_list. This should ideally be run directly through mongo shell
    for large collections.

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        collection      (str): Name of collection to use.
        lang_list   (list): List of languages to match on.
        output_name (str): Name of the collection to store ids of non removed tweets.
    """
    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]
    dbo = client[db_name]
    bulk = dbo[collection].initialize_unordered_bulk_op()
    count = 0

    pipeline = [
        {"$match": {"lang": {"$nin": lang_list}}},
        {"$project": {"lang": 1, "_id": 1}},
        {"$group": {
            "_id": {
                "lang": "$lang",
            },
            "ids": {"$push": "$_id"}
        }},
        {"$project": {"ids": 1}}
    ]
    cursor = dbo[collection].aggregate(pipeline, allowDiskUse=True)
    print("Finished aggregation. Iterating now")

    for document in cursor:
        bulk.find({"_id": {"$in": document["ids"]}}).remove()
        count = count + 1
        print("Count:", count)

        if count % 1000 == 0:
            print("Running bulk execute")
            bulk.execute()
            bulk = dbo[collection].initialize_unordered_bulk_op()

    if count % 1000 != 0:
        print("Running bulk execute")
        bulk.execute()

    pipeline = [
        {"$project": {"_id": 1}},
        {"$out": output_name}
    ]
    dbo[collection].aggregate(pipeline, allowDiskUse=True)
