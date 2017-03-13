import json
import requests


class EmotionDetection(object):
    """docstring for EmotionDetection"""

    def __init__(self):
        super(EmotionDetection, self).__init__()

    __url = 'http://140.114.77.14:8080/webresources/jammin/emotion'
    #__url = 'http://192.168.2.30:8080/webresources/jammin/emotion'
    __emo = None

    def get_obj(self, text, lang='en'):
        payload = {"text": text, "lang": lang}
        r = requests.post(self.__url,  data=json.dumps(payload))
        return r

    # order: 0 the first emotion and 1 the second emotion if it has the second
    # one
    def get_emotion(self, text, order=0, lang='en'):
        r = self.get_obj(text, lang)
        return r.json()["groups"][order]["name"]

    def get_emotion_json(self, text, lang='en'):
        r = self.get_obj(text, lang)
        self.__emo = r.json()
        return self.__emo

    def is_ambigous(self, text, lang='en'):
        r = self.get_obj(text, lang)
        r = r.json()

        if r['ambiguous'] == 'no':
            return False
        else:
            return True

    def is_ambigous_json(self, r):
        if r['ambiguous'] == 'no':
            return False
        else:
            return True
