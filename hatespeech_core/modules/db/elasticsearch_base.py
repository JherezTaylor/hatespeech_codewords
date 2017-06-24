# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module stores base methods for elasticsearch.
"""

from pymongo import InsertOne
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from ..utils import settings
from ..db import mongo_base


def connect(es_url=None):
    """Initializes an elasticseach conection object.

    Returns:
        elaticsearch.Elasticsearch: Connection object for Elasticsearch
    """
    if es_url is None:
        es_url = settings.ES_URL
    try:
        es_host = {"host": es_url, "port": 9200}
        _es = Elasticsearch([es_host])
        if _es.ping():
            settings.logger.info(
                "Connected to ElasticSearch at %s successfully", es_url)
    except ValueError as ex:
        settings.logger.error(
            "Could not connect to ElasticSearch: %s", ex, exc_info=True)
    return _es


def more_like_this(_es, es_index, field, like_list, min_term_freq, max_query_terms):
    """Build and execute a more like this query on the like document
    See https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-mlt-query.html

    Returns
        result (list): list of documents that match the like document.
    """
    queries = [{
        "stored_fields": field,
        "query": {
            "more_like_this": {
                "fields": field,
                "like": like,
                "min_term_freq": min_term_freq,
                "max_query_terms": max_query_terms
            }
        }
    } for like in like_list]

    results = []
    for query in queries:
        res = _es.search(index=es_index, body=query)
        results.append([hit['fields'][field[0]][0]
                        for hit in res['hits']['hits']])
    return results


def match(_es, es_index, doc_type, field, lookup_list):
    """Build and execute an exact match query, matching on the passed field and the
    values in the lookup_list.

    Returns
        result (list): list of documents.
    """
    query = {
        "query": {
            "constant_score": {
                "filter": {
                    "terms": {
                        field: lookup_list
                    }
                }
            }
        }
    }
    results = es_helpers.scan(
        _es, index=es_index, doc_type=doc_type, query=query, scroll='2m', size=3400)
    return results


def aggregate(_es, es_index, field, use_range, query_filter, size=10, min_doc_count=10):
    """Build and execute an aggregate query, matching on the passed field.
    Args:
        use_range (bool): Apply time range to query.
        query_filter (str): Documents must match the requirements in this
                query, formatted with ELS Query DSL. example: "_exists_:hs_keyword_matches"
                Pass "*" to match all documents.
    Returns
        result (list): list of documents.
    """

    if use_range:
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "query_string": {
                                "query": query_filter,
                                "analyze_wildcard": True
                            }
                        },
                        {
                            "range": {
                                "created_at": {
                                    "gte": 1339553696602,
                                    "lte": 1497320096602,
                                    "format": "epoch_millis"
                                }
                            }
                        }
                    ],
                    "must_not": []
                }
            },
            "size": 0,
            "_source": {
                "excludes": []
            },
            "aggs": {
                "2": {
                    "terms": {
                        "field": field,
                        "size": size,
                        "order": {
                            "_count": "desc"
                        },
                        "min_doc_count": min_doc_count
                    }
                }
            }
        }
    else:
        query = {
            "query": {
                "query_string": {
                    "query": query_filter,
                    "analyze_wildcard": True
                }
            },
            "size": 0,
            "_source": {
                "excludes": []
            },
            "aggs": {
                "2": {
                    "terms": {
                        "field": field,
                        "size": size,
                        "order": {
                            "_count": "desc"
                        },
                        "min_doc_count": min_doc_count
                    }
                }
            }
        }
    response = _es.search(index=es_index, body=query)

    results = {item["key"]: item["doc_count"]
               for item in response["aggregations"]["2"]["buckets"]}
    if not results:
        return response
    else:
        return results, response["hits"]["total"]


def count(_es, es_index, query):
    """Execute a query and get the number of matches for that query.

    Returns
      result (int): Query count result.
    """
    response = _es.count(index=es_index, body=query)
    return response["count"]


def get_els_subset_size(_es, es_index, field):
    """ Return both the number of documents that have the given field and the inverse.
    """
    positive_query = {
        "query": {
            "exists": {
                "field": field
            }
        }
    }

    negative_query = {
        "query": {
            "bool": {
                "must_not": {
                    "exists": {
                        "field": field
                    }
                }
            }
        }
    }

    positive_count = count(_es, es_index, positive_query)
    negative_count = count(_es, es_index, negative_query)

    result = {}
    result["positive_count"] = positive_count
    result["negative_count"] = negative_count
    return result


def migrate_es_tweets(connection_params, args):
    """ Scroll an elasticsearch instance and insert the tweets into MongoDB
    """

    db_name = connection_params[0]
    target_collection = connection_params[1]

    es_url = args[0]
    es_index = args[1]
    doc_type = args[2]
    field = args[3]
    lookup_list = args[4]
    _es = connect(es_url)

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    es_results = match(
        _es, es_index, doc_type, field, lookup_list)

    operations = []
    for doc in es_results:
        operations.append(InsertOne(doc["_source"]))

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if operations:
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
