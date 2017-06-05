# Automatic Hate Speech Detection: Preprocessing

This repo contains my initial scripts for playing with the data.

## Background

Merriam-Webster defines a Coded Word as *a word or phrase that has a secret meaning or that is used instead of another word or phrase to avoid speaking directly*.

*Hate speech is anything that is considered to incite violence, attacks or insults a person or group on the basis of ethnic, gender, color, religion, sexual orientation or disability.*

Social networks have built tools to automatically detect, filter or block these posts but there are still gaps that lead to serious consequences

The aim of my proposal is to identify the patterns associated with the use of hate speech and coded words in an effort to account for when new words are introduced to the hate corpus and to minimize the reliance on a dictionaries.


Here I run some API routes from the [Hatebase API](https://www.hatebase.org/connect_api), an account is required. This is a nice corpus of hate related words in a number of languages along with alternate words and Twitter sightings etc. My first task was to get a dictionary of words that I can crawl Twitter with.

API calls take the format of:

> http://api.hatebase.org/version/key/query-type/output/encoded-filters

A real word example takes the following format:
> https://api.hatebase.org/v3-0/API_KEY/vocabulary/json/about_nationality=1%7clanguage=eng%7cpage=1

The list of available filters can be found at the Hatebase API link mentioned above.  

Unzip dependency2vec to `hatespeech_core/data/conll_data`