import regex
import pandas as pd
import numpy as np

class PatternVectorizer:

  def __init__(self, patterns, binary=False):
    self.binary = binary
    
    vocabulary = pd.DataFrame()
    vocabulary['patterns'] = patterns
    vocabulary['regex'] = vocabulary.patterns.apply(
      lambda p: regex.compile(PatternVectorizer.pattern_to_regexp(p))
    )  
    self.vocabulary = vocabulary
    
  def transform(self, documents):
    X = np.array([*map(lambda doc: self.count_vocab(doc), documents)], dtype=np.int32)
    if self.binary:
      X[X>0] = 1
    return X
    
  def count_vocab(self, text):
    return self.vocabulary.regex.apply(lambda voc: len(voc.findall(text)))    
  
  @classmethod
  def token_to_regexp(cls, token):
    tok_to_reg = {
      '.+': "((?![@,#])[\\p{L}\\p{M}*\\p{N}_]+|(?![@,#])\\p{Punct}+)",
      '<hashtag>': "#([\\p{L}\\p{M}*\\p{N}_]+|(?![@,#])\\p{Punct}+)",
      '<usermention>': "@([\\p{L}\\p{M}*\\p{N}_]+|(?![@,#])\\p{Punct}+)",
      '<url>': "http://([\\p{L}\\p{M}*\\p{N}_\\.\\/]+|(?![@,#])\\p{Punct}+)"
    }
    return tok_to_reg.get(token) or token
  
  @classmethod
  def pattern_to_regexp(cls, pattern_str):
    delimRegex = "((?![@,#])\\b|\\p{Z}+|$|^|(?![@,#])\\p{Punct})"
    patt = pattern_str.strip()
    tokens = patt.split(" ")
    tokens_reg = map(lambda t: cls.token_to_regexp(t),tokens)
    pattern = delimRegex + delimRegex.join(tokens_reg) + delimRegex
    return pattern
