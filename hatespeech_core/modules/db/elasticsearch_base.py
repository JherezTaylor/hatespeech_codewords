from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from ..utils import settings


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
        settings.logger.error("Could not connect to ElasticSearch: %s", ex)
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
