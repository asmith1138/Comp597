import numpy as np
from PIL import Image
from wordcloud import WordCloud
import twitter

frequency = {}

keyword = input("Enter a keyword for search: ") 
count = input("Enter how many tweets do you want to return: ") 

#Replace with keyword twitter search
document_text = open('test.txt', 'r')
text_string = document_text.read().lower()


#twitter code
api = twitter.Api(consumer_key='yXtJWE2Gtx4tooNgFwAsXFgFG',
                  consumer_secret='a42mqHrz1pKFYCvbXBTCxR9WJghlSmvBe4vbJllQ8QE4FPl8ko',
                  access_token_key='1353080476387446785-MT1vtnHxOW4txaDfqKoLUvgm9xOYHZ',
                  access_token_secret='cW3tCyEwXIE0Jw0KgffrsH7aAEMJypdwX5iMLwhIGczp9')

results = api.GetSearch(
    raw_query=f"q={keyword}&result_type=recent&count={count}")

#map text to string
def mapText(lst):
    return lst.text

str1 = " "
tweets = str1 + str1.join(list(map(mapText,results)))

#updated pattern to remove non words
#'(?<=[\s\"\'\(\[])\b[a-zA-z\'\-]+\b(?=[\s]|.\s)' This is the final pattern
#'\b[a-z\'-]+\b' This is the original pattern
#idea for this package came from this article: 
# https://www.askpython.com/python/examples/word-cloud-using-python#:~:text=%20How%20to%20Create%20a%20Word%20Cloud%20using,cloud%20mask%20and%20set%20stop%20words%20More%20

def create_wordcloud(text):
    #Source: https://www.onlinewebfonts.com/icon/5830
    imagemask = np.array(Image.open("cloud.png"))

    stopwordsFile = open('stopwords.txt', 'r')
    stopwordsString = stopwordsFile.read().lower().split('\n')
    stopwords = set(stopwordsString)
 
    wc = WordCloud(contour_color="white",
                    max_words=1000000000, 
                    mask=imagemask,
                    regexp=r"(?<=[\s\"\'\(\[])\b[a-zA-Z\'\-]+\b(?=[\s]|.\s)",
                    stopwords=stopwords,
                    min_word_length=3)
     
    wc.generate(text)
    wc.to_file("output.png")

create_wordcloud(tweets.lower())
