import csv
import math
import random
import sys


class Advertiser:
    def __init__(self, id, keyword, bidValue, budget):
        self.id = id
        self.keyword = keyword
        self.bidValue = bidValue
        self.budget = budget


# Read queries and create dictionaries for advertisers and queries
def readData():
    queryNeighbors = dict()
    advertiserBudgets = dict()
    advertisersObjects = dict()
    with open('bidder_dataset.csv', 'r') as bidderDataset:
        bidderReader = csv.reader(bidderDataset, delimiter=',')
        next(bidderReader)
        for row in bidderReader:
            if row[3] != '':
                advertiserBudgets[int(row[0])] = float(row[3])
            if int(row[0]) not in advertisersObjects:
                advertisersObjects[int(row[0])] = Advertiser(int(row[0]), [row[1]], [float(row[2])],
                                                             advertiserBudgets[int(row[0])])
            else:
                advertisersObjects[int(row[0])].keyword.append(row[1])
                advertisersObjects[int(row[0])].bidValue.append(float(row[2]))
            if row[1] not in queryNeighbors:
                queryNeighbors[row[1]] = [(int(row[0]))]
            else:
                queryNeighbors[row[1]].append(int(row[0]))
    return (queryNeighbors, advertiserBudgets, advertisersObjects)


# Check if the advertiser has sufficient budget to bid for the query or not
def isNeighborAvailable(query, qNeighbors, advertisersObjects):
    for neighbor in qNeighbors:
        bidValue = advertisersObjects[neighbor].bidValue[advertisersObjects[neighbor].keyword.index(query)]
        if advertisersObjects[neighbor].budget >= bidValue:
            return True
    return False


# Greedy approach
def greedy(queries, queryNeighbors, advertiserBudgets, advertisersObjects):
    revenue = 0
    for query in queries:
        qNeighbors = queryNeighbors[query]
        if isNeighborAvailable(query, qNeighbors, advertisersObjects):
            highestBid = 0
            highestBidNeighbor = qNeighbors[0]
            for neighbor in qNeighbors:
                bidValue = advertisersObjects[neighbor].bidValue[advertisersObjects[neighbor].keyword.index(query)]
                if bidValue > highestBid and advertisersObjects[neighbor].budget >= bidValue:
                    highestBid = bidValue
                    highestBidNeighbor = neighbor
            advertisersObjects[highestBidNeighbor].budget -= highestBid
            revenue += highestBid
    return revenue


# MSVV Approach
def msvv(queries, queryNeighbors, advertiserBudgets, advertisersObjects):
    revenue = 0
    for query in queries:
        qNeighbors = queryNeighbors[query]
        if isNeighborAvailable(query, qNeighbors, advertisersObjects):
            highestValue = -sys.maxsize
            highestValueNeighbor = qNeighbors[0]
            for neighbor in qNeighbors:
                Xu = (advertiserBudgets[neighbor] - advertisersObjects[neighbor].budget) / advertiserBudgets[neighbor]
                bidValue = advertisersObjects[neighbor].bidValue[advertisersObjects[neighbor].keyword.index(query)]
                if (1 - math.exp(Xu - 1)) * bidValue > highestValue and advertisersObjects[neighbor].budget >= bidValue:
                    highestValue = (1 - math.exp(Xu - 1)) * bidValue
                    highestValueNeighbor = neighbor
                    highestBid = bidValue
            advertisersObjects[highestValueNeighbor].budget -= highestBid
            revenue += highestBid
    return revenue


# Balance Approach
def balance(queries, queryNeighbors, advertiserBudgets, advertisersObjects):
    revenue = 0
    for query in queries:
        qNeighbors = queryNeighbors[query]
        if isNeighborAvailable(query, qNeighbors, advertisersObjects):
            highestUnspentBudget = -sys.maxsize
            highestUnspentBudgetNeighbor = qNeighbors[0]
            for neighbor in qNeighbors:
                bidValue = advertisersObjects[neighbor].bidValue[advertisersObjects[neighbor].keyword.index(query)]
                if advertisersObjects[neighbor].budget > highestUnspentBudget and advertisersObjects[
                    neighbor].budget >= bidValue:
                    highestUnspentBudget = advertisersObjects[neighbor].budget
                    highestUnspentBudgetNeighbor = neighbor
                    highestBid = bidValue
            advertisersObjects[highestUnspentBudgetNeighbor].budget -= highestBid
            revenue += highestBid
    return revenue


# Main function
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python adwords.py method_name")
        print("method_name can be either 'greedy', 'msvv' or 'balance'")
        exit()

    random.seed(0)
    revenue = 0
    totalRevenue = 0
    averageRevenue = 0
    with open('queries.txt', 'r') as q:
        queries = [query.strip() for query in q]

    queryNeighbors, advertiserBudgets, advertisersObjects = readData()
    if sys.argv[1] == 'greedy':
        totalRevenue = greedy(queries, queryNeighbors, advertiserBudgets, advertisersObjects)
        for i in range(100):
            queryNeighbors, advertiserBudgets, advertisersObjects = readData()
            tempQueries = queries
            random.shuffle(tempQueries)
            revenue += greedy(tempQueries, queryNeighbors, advertiserBudgets, advertisersObjects)
        averageRevenue = revenue / 100
    elif sys.argv[1] == 'msvv':
        totalRevenue = msvv(queries, queryNeighbors, advertiserBudgets, advertisersObjects)
        for i in range(100):
            queryNeighbors, advertiserBudgets, advertisersObjects = readData()
            tempQueries = queries
            random.shuffle(tempQueries)
            revenue += msvv(tempQueries, queryNeighbors, advertiserBudgets, advertisersObjects)
        averageRevenue = revenue / 100
    elif sys.argv[1] == 'balance':
        totalRevenue = balance(queries, queryNeighbors, advertiserBudgets, advertisersObjects)
        for i in range(100):
            queryNeighbors, advertiserBudgets, advertisersObjects = readData()
            tempQueries = queries
            random.shuffle(tempQueries)
            revenue += balance(tempQueries, queryNeighbors, advertiserBudgets, advertisersObjects)
        averageRevenue = revenue / 100
    else:
        print('Invalid method. Please try again.')
        exit()

    print('Revenue:', round(totalRevenue, 2))
    optimalMatching = sum(advertiserBudgets.values())
    competitiveRatio = averageRevenue / optimalMatching
    print('Competitive Ratio:', round(competitiveRatio, 2))
