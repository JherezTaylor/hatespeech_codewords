// Transporter pipeline file

// Uncomment the t.Source as required, only one can be active at a time
// Yes, I'm aware of the string

var field_list = ["_id", "preprocessed_txt", "created_at", "hs_keyword_matches", "dependencies", "feat_dep_root_word",
  "feat_dep_pos_rootPos", "feat_dep_unigrams", "feat_dep_trigrams", "comment_length", "unknown_words", "hashtags", "tokens",
  "annotation", "brown_cluster_ids"
]

var source_annotations = mongodb({
  "uri": "mongodb://runner:74TXc0WF3luv@127.0.0.1:27017/twitter_annotated_datasets?authSource=admin",
  "timeout": "30s",
  // "tail": false,
  // "wc": 1,
  // "fsync": false,
  "bulk": true
  // "collection_filters": "{}"
})

var sink_naacl = elasticsearch({
  "uri": "http://127.0.0.1:9200/naacl_16",
  "timeout": "30s"
})

// t.Source("source", source_annotations, "/^NAACL_SRW_2016$/").Transform(js({
//   "filename": "prep_data.js"
// })).Transform(pick({
//   "fields": field_list
// })).Save("sink", sink_naacl)

var sink_nlp_css_16 = elasticsearch({
  "uri": "http://127.0.0.1:9200/nlp_css_16",
  "timeout": "30s"
})

// t.Source("source", source_annotations, "/^NLP_CSS_2016_expert$/").Transform(js({
//   "filename": "prep_data.js"
// })).Transform(pick({
//   "fields": field_list
// })).Save("sink", sink_nlp_css_16)

var sink_crwdflr = elasticsearch({
  "uri": "http://127.0.0.1:9200/crwdflr",
  "timeout": "30s"
})

// t.Source("source", source_annotations, "/^crowdflower$/").Transform(js({
//   "filename": "prep_data.js"
// })).Transform(pick({
//   "fields": field_list
// })).Save("sink", sink_crwdflr)

var source_twitter = mongodb({
  "uri": "mongodb://runner:74TXc0WF3luv@127.0.0.1:27017/twitter?authSource=admin",
  "timeout": "30s",
  "bulk": true
})

var sink_melvyn_hs = elasticsearch({
  "uri": "http://127.0.0.1:9200/melvyn_hs",
  "timeout": "30s"
})

// t.Source("source", source_twitter, "/^melvyn_hs_users$/").Transform(js({
//   "filename": "prep_data.js"
// })).Transform(pick({
//   "fields": field_list
// })).Save("sink", sink_melvyn_hs)

var sink_core_tweets = elasticsearch({
  "uri": "http://127.0.0.1:9200/core_tweets",
  "timeout": "30s"
})

// t.Source("source", source_twitter, "/^tweets$/").Transform(js({
//   "filename": "prep_data.js"
// })).Transform(pick({
//   "fields": field_list
// })).Save("sink", sink_core_tweets)

var source_manchester = mongodb({
  "uri": "mongodb://runner:74TXc0WF3luv@127.0.0.1:27017/manchester_event?authSource=admin",
  "timeout": "30s",
  "bulk": true
})

var sink_manchester = elasticsearch({
  "uri": "http://127.0.0.1:9200/manchester_event",
  "timeout": "30s"
})

// t.Source("source", source_manchester, "/^tweets$/").Transform(js({
//   "filename": "prep_data.js"
// })).Transform(pick({
//   "fields": field_list
// })).Save("sink", sink_manchester)

var source_unfiltered = mongodb({
  "uri": "mongodb://runner:74TXc0WF3luv@127.0.0.1:27017/unfiltered_stream_May17?authSource=admin",
  "timeout": "30s",
  "bulk": true
})

var sink_unfiltered = elasticsearch({
  "uri": "http://127.0.0.1:9200/unfiltered_stream",
  "timeout": "30s"
})

// t.Source("source", source_unfiltered, "/^tweets$/").Transform(js({
//   "filename": "prep_data.js"
// })).Transform(pick({
//   "fields": field_list
// })).Save("sink", sink_unfiltered)


// t.Source("source", source, "/.*/").Save("sink", sink, "/.*/")