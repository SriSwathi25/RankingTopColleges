import re
import tweepy
import csv
import nltk
import pandas as pd
import string
import math
from tweepy import OAuthHandler 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import snscrape.modules.twitter as sntwitter

import TwitterCredentials
nltk.download('wordnet')
nltk.download('vader_lexicon')

class Twitter_Main():
    def __init__(self):
        
        consumer_key = ""
        consumer_secret = ""
        access_token = ""
        access_token_secret = ""
        try:
            self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth,wait_on_rate_limit=False)
        except:
            print("\nInvalid Authentication")
        
        #Stopwords file opening
        with open('stopwords_file.txt', 'r') as f:
            k=f.readlines()
        self.stoplist=[k[i][:-1] for i in range(len(k))]
    
    def removePostfix(self,argWord):
        #stemming
        leaves = ["s", "es", "ed", "er", "ly", "ing"]
        for leaf in leaves:
            if argWord[-len(leaf):] == leaf:
                return argWord[:-len(leaf)]
            
        return argWord
        
            
    def preprocessing(self,tweet):
        
        #Convert to lower case
        tweet = tweet.lower()
        #Removing www.* or https?://*  or @*
        tweet = re.sub('www\.[^\s]+',' ',tweet)
        tweet=re.sub("https?://[^\s]+",' ',tweet)
        tweet= re.sub("@[^\s]+",' ' , tweet)
        #Remove rt Retweet
        tweet=re.sub("rt",' ',tweet)
        #Remove #word 
        tweet = re.sub(r'#([^\s]+)', ' ', tweet)
        #Remove punctuations
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        tweet=regex.sub('', tweet)
        #Only English words
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)
        #tokenize or split in words
        tweet = tweet.split()
        #Remove duplicate words and extra spaces 
        tweet=list(set(tweet))
        if ' ' in tweet:
            tweet.remove(' ')

            
        tweet1=[]
        
        #Lemmatizing
        wordnet_lemmatizer = WordNetLemmatizer()
        for i in range(len(tweet)):
            if tweet[i] not in self.stoplist:
                #Stemming
                c=self.removePostfix(tweet[i])
                t=wordnet_lemmatizer.lemmatize(c)
                tweet1.append(t)
                    
         
        tweet1 = ' '.join(tweet1)
        return tweet1 
    
   
        
    

 
# creating object of Twitter_Main Class 
api = Twitter_Main() 
# calling function to get tweets 
k1=input("Enter the query:")
k2=input("Enter the name of file with tweets :");
k2=k2+".csv"
csvFile = open(k2, 'a') #creates a file in which you want to store the data.
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["Tweet", "Sentiment"])
maxTweets = 2500  # the number of tweets you require
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(k1 + " since:2000-01-01 until:2021-01-03").get_items()) :
        if i > maxTweets :
            break
        #print(tweet.date)
        csvWriter.writerow([tweet.content.encode('utf-8')])

print("\n---Process is over\n")

            
