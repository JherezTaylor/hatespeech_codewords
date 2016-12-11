// Remove unwanted languages
lang_list = ['en', 'und'];

db.tweets.bulkWrite(
    [
        {
            deleteMany: 
            {
                "filter": {"lang": {$nin: lang_list}}
            }
        }
    ],
    { ordered: false }
)

// Remove retweets
db.tweets.bulkWrite(
    [
        {
            deleteMany: 
            {
                "filter": {"retweet_count": {$gt: 0}}
            }
        }
    ],
    { ordered: false }
)


// Find and compress entities.urls
var bulk = db.tweets.initializeOrderedBulkOp(),
    count = 0;

db.tweets.aggregate([
    {$project: {"entities.urls": 1, _id:1}},
    {$unwind:"$entities.urls"},
    {$group:{_id:'$_id',urls:{$addToSet:'$entities.urls.expanded_url'}}}
],{ "allowDiskUse": true}).forEach(function(doc) {
    bulk.find({ "_id": doc._id } ).update({$unset: {"entities.urls": ""}, $set: {"urls" : doc.urls}}); 
    count++;

    // Execute 1 in 1000 and re-init
    if ( count % 1000 == 0 ) {
       bulk.execute();
       bulk = db.tweets.initializeOrderedBulkOp();
    }
});

if ( count % 1000 != 0 ) 
    bulk.execute();

// Find and compress entities.user_mentions
var bulk = db.tweets.initializeOrderedBulkOp(),
    count = 0;

db.tweets.aggregate([
    {$project: {"entities.user_mentions": 1, _id:1}},
    {$unwind:"$entities.user_mentions"},
    {$group:{_id:'$_id', user_mentions:{$addToSet:"$entities.user_mentions.screen_name"}, user_mentions_id_str:{$addToSet:"$entities.user_mentions.id_str"}}}
],{ "allowDiskUse": true}).forEach(function(doc) {
    bulk.find({ "_id": doc._id } ).update({$unset: {"entities.user_mentions": ""}, $set: {"user_mentions" : doc.user_mentions, "user_mentions_id_str" : doc.user_mentions_id_str}}); 
    count++;

    // Execute 1 in 1000 and re-init
    if ( count % 1000 == 0 ) {
       bulk.execute();
       bulk = db.tweets.initializeOrderedBulkOp();
    }
});

if ( count % 1000 != 0 ) 
    bulk.execute();

// Find and compress entities.hashtags
var bulk = db.tweets.initializeOrderedBulkOp(),
    count = 0;
    
db.tweets.aggregate([
    {$project: {"entities.hashtags": 1, _id:1}},
    {$unwind:"$entities.hashtags"},
    {$group:{_id:'$_id', hashtags:{$addToSet:"$entities.hashtags.text"} }}
],{ "allowDiskUse": true}).forEach(function(doc) {
    bulk.find({ "_id": doc._id } ).update({$unset: {"entities.hashtags": ""}, $set: {"hashtags" : doc.hashtags}}); 
    count++;

    // Execute 1 in 1000 and re-init
    if ( count % 1000 == 0 ) {
       bulk.execute();
       bulk = db.tweets.initializeOrderedBulkOp();
    }
});

if ( count % 1000 != 0 ) 
    bulk.execute();

// Find and compress entities.media
var bulk = db.tweets.initializeOrderedBulkOp(),
    count = 0;
    
db.tweets.aggregate([
    {$project: {"entities.media": 1, _id:1}},
    {$unwind:"$entities.media"},
    {$group:{_id:'$_id', media:{$push:{"url": "$entities.media.media_url", "type": "$entities.media.type"}} }}
],{ "allowDiskUse": true}).forEach(function(doc) {
    bulk.find({ "_id": doc._id } ).update({$unset: {"entities.media": ""}, $set: {"media" : doc.media}}); 
    count++;

    // Execute 1 in 1000 and re-init
    if ( count % 1000 == 0 ) {
       bulk.execute();
       bulk = db.tweets.initializeOrderedBulkOp();
    }
});

if ( count % 1000 != 0 ) 
    bulk.execute();

// Remove unwanted fields
db.tweets.bulkWrite(
    [
        {
            updateMany:
            {
                "filter": {},
                "update": {
                    $unset: {contributors: "", truncated: "","retweet_count": "", retweeted: "", display_text_range: "", 
                        retweeted_status: "", extended_entities:"",
                        entities: "", favorited: "", id: "", "user.follow_request_sent": "",
                        "user.profile_use_background_image": "", "user.default_profile_image": "",
                        "user.profile_sidebar_fill_color": "","user.profile_image_url_https": "",
                        "user.profile_sidebar_border_color": "","user.profile_text_color": "",
                        "user.profile_sidebar_border_color": "","user.id": "",
                        "user.profile_background_color": "","user.profile_background_image_url_https": "",
                        "user.profile_link_color": "","user.profile_image_url": "",
                        "user.profile_background_image_url": "","user.profile_background_tile": "",
                        "user.notifications": "","user.default_profile": "",
                        "user.is_translator": ""

                    },
                    $set: {"preprocessed": true}
                },
                "upsert": false
            }
        }
    ],
    { ordered: false }
)

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