from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt

def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)

def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    
    #counts look like this [[('positive',200), ('negative', 100)], [('positive',300), ('negative', 200)], ...]
    posCounts = [x[0][1] for x in counts if x]
    negCounts = [x[1][1] for x in counts if x]

    #Plot labeling and plotting
    plt.plot(posCounts,'-o', label = 'positive')
    plt.plot(negCounts,'-o', label = 'negative')
    plt.ylabel('Word count')
    plt.xlabel('Time step')
    plt.legend(loc='upper left')
    plt.axis([-1, 12, 0, 300])
    plt.show()


def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """

    #Open the file, split it on the newline, trim the word
    words = open(filename).read().strip().split('\n')
    return words
    
def updateFunc(newSum, oldSum):
    #newSum + oldSum
    #For the first time step, oldSum will be zero as there is no count of words
    return newSum[0] + (oldSum or 0)
    
def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    
    #Split the tweet at newline and collect the words into a flatmap
    words = tweets.flatMap(lambda line: line.split(" ")).map(lambda word: word.encode("ascii", "ignore").lower())
    
    #According to their sentiment i.e. positive/negative, create a tuple for each word
    pairs = words.map(lambda word: ("positive", 1) if word in pwords else (("negative", 1) if word in nwords else (word, 1)))
    
    #Filter out the words which are not in either sentiment i.e. they are treated as neural
    pairs = pairs.filter(lambda x: x[0] == 'positive' or x[0] == 'negative')

    #This RDD will contain the count of the positive and negative words just for the current time step
    wordCounts = pairs.reduceByKey(lambda x, y: x + y)

    #This RDD will contain the running count of the total positive and negative words
    totalCounts = wordCounts.updateStateByKey(updateFunc)

    #Stores the count with all the previous values as well 
    counts = []    
    totalCounts.pprint()
    
    #Append the new count to the counts container
    wordCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)
    
    return counts


if __name__=="__main__":
    main()
