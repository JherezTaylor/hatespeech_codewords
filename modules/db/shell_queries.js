// Remove unwanted languages
lang_list = ['en', 'und'];

db.tweets.bulkWrite([
   { deleteMany : { "filter" : { "lang": { $nin : lang_list } } } }
] )


db.tweets.find({$text: {$search:"trump"}}).limit(5)
db.tweets.count({"lang":"en"})
db.tweets.distinct("lang")

db.tweets.aggregate([ { $match: { 'lang': {$in: ['en','und']} } }, { $out: "dataset" } ],  {allowDiskUse:true})
db.tweets.aggregate([ { $match: { 'lang': {$in: ['und']} } }, {'$limit' : 500}, { $out: "und_backup" } ])
db.tweets.aggregate([{$match : {lang : "en"}},{$group : { _id : "$lang", total : { $sum : 1 } }}])

// -----  Top K Hashtags ----- //

// English
db.tweets.aggregate([{$match: {'lang': {$in: ['en']}}}, {$project: {'entities.hashtags': 1, _id : 0}}, {$unwind: '$entities.hashtags'}, {$group: {_id: '$entities.hashtags.text', count: {$sum: 1}}}, {$sort: {count: -1}}, {$project: {"hashtag": "$_id", "count": 1, "_id": 0}}, { $out : "hashtag_dist_en" }])

// Undefined
db.tweets.aggregate([{$match: {'lang': {$in: ['und']}}}, {$project: {'entities.hashtags': 1, _id : 0}}, {$unwind: '$entities.hashtags'}, {$group: {_id: '$entities.hashtags.text', count: {$sum: 1}}}, {$sort: {count: -1}}, {$project: {"hashtag": "$_id", "count": 1, "_id": 0}}, { $out : "hashtag_dist_und" }])

// Spanish
db.tweets.aggregate([{$match: {'lang': {$in: ['es']}}}, {$project: {'entities.hashtags': 1, _id : 0}}, {$unwind: '$entities.hashtags'}, {$group: {_id: '$entities.hashtags.text', count: {$sum: 1}}}, {$sort: {count: -1}}, {$project: {"hashtag": "$_id", "count": 1, "_id": 0}}, { $out : "hashtag_dist_es" }])

db.tweets.aggregate([{$match: {'lang': {$in: ['en']}}}, {$project: {'entities.hashtags': 1, _id : 0}}, {$unwind: '$entities.hashtags'}, {$group: {_id: '$entities.hashtags.text', count: {$sum: 1}}}, {$sort: {count: -1}}], {explain:true})

// -----  Top K User Mentions ----- //

// English
db.tweets.aggregate([{$match: {'lang': {$in: ['en']}}}, {$project: {'entities.user_mentions': 1, _id: 0}}, {$unwind: '$entities.user_mentions'}, {$group: {_id: {id_str: '$entities.user_mentions.id_str', 'screen_name': '$entities.user_mentions.screen_name'}, count: {$sum: 1}}}, {$project: {id_str: '$_id.id_str', 'screen_name': '$_id.screen_name', 'count': 1, '_id': 0}}, {$sort: {count: -1}}, { $out : "user_mentions_dist_en" }], {allowDiskUse:true})

// Undefined
db.tweets.aggregate([{$match: {'lang': {$in: ['und']}}}, {$project: {'entities.user_mentions': 1, _id: 0}}, {$unwind: '$entities.user_mentions'}, {$group: {_id: {id_str: '$entities.user_mentions.id_str', 'screen_name': '$entities.user_mentions.screen_name'}, count: {$sum: 1}}}, {$project: {id_str: '$_id.id_str', 'screen_name': '$_id.screen_name', 'count': 1, '_id': 0}}, {$sort: {count: -1}}, { $out : "user_mentions_dist_und" }], {allowDiskUse:true})

// Spanish
db.tweets.aggregate([{$match: {'lang': {$in: ['es']}}}, {$project: {'entities.user_mentions': 1, _id: 0}}, {$unwind: '$entities.user_mentions'}, {$group: {_id: {id_str: '$entities.user_mentions.id_str', 'screen_name': '$entities.user_mentions.screen_name'}, count: {$sum: 1}}}, {$project: {id_str: '$_id.id_str', 'screen_name': '$_id.screen_name', 'count': 1, '_id': 0}}, {$sort: {count: -1}}, { $out : "user_mentions_dist_es" }], {allowDiskUse:true})