import csv
from igraph import *
from scipy import spatial

g = Graph()
attributes = []


def readUsers():
    with open('data/fb_caltech_small_attrlist.csv') as f:
        csvReader = csv.reader(f, delimiter=',')
        header = next(csvReader)
        for attr in header:
            attributes.append(attr)

        userId = 0
        for user in csvReader:
            g.add_vertex(userId)
            for attributeID in range(0, len(attributes)):
                g.vs[userId][attributes[attributeID]] = float(user[attributeID])
            userId += 1


def readEdges():
    with open('data/fb_caltech_small_edgelist.txt') as f:
        lines = f.readlines()
        for edge in lines:
            edge = edge.strip().split(' ')
            source = int(edge[0])
            destination = int(edge[1])
            g.add_edge(source, destination)
            g.es[g.get_eid(source, destination)]["weight"] = 1


def sumOfEdgeWeightsForVertex(vertex):
    sumOfEdges = 0
    currentEdges = g.incident(vertex)
    for edge in currentEdges:
        sumOfEdges += g.es[edge]["weight"]
    return sumOfEdges


def calculateQNewman(setOfVertices, currentVertex):
    communityDegree = 0
    links = 0
    numOfEdges = g.ecount()
    for vertex in setOfVertices:
        if g.are_connected(currentVertex, vertex):
            links += g.es[g.get_eid(currentVertex, vertex)]["weight"]
    currentVertexDegree = sumOfEdgeWeightsForVertex(currentVertex)
    for vertex in setOfVertices:
        communityDegree += sumOfEdgeWeightsForVertex(vertex)
    QNewman = (1.0 / (2 * numOfEdges)) * (links - ((float(currentVertexDegree) / (2 * numOfEdges)) * communityDegree))
    return QNewman


def sac1(alpha):
    numOfVertices = g.vcount()
    numOfEdges = g.ecount()
    communities = dict()
    for communityIndex in range(0, numOfVertices):
        communities[communityIndex] = [communityIndex]
    cosineSimilarityMatrix = [[0 for x in range(numOfVertices)] for y in range(numOfVertices)]
    for idA in xrange(numOfVertices):
        for idB in xrange(idA, numOfVertices):
            attrA = []
            attrB = []
            for attribute in attributes:
                attrA.append(g.vs[idA][attribute])
                attrB.append(g.vs[idB][attribute])
            cosineSimilarityMatrix[idA][idB] = 1 - spatial.distance.cosine(map(float, attrA), map(float, attrB))
            cosineSimilarityMatrix[idB][idA] = cosineSimilarityMatrix[idA][idB]
    if numOfEdges > 0:
        vertexList = list(range(0, numOfVertices))
        for phaseOneIteration in range(0, 15):
            print "\t Phase One iteration: ", phaseOneIteration + 1
            convergence = True
            for currentVertex in vertexList:
                communityWithMaxGain = -1
                maxCompositeModularityGain = 0
                originalCommunity = -1
                for community in communities.keys():
                    if currentVertex in communities[community]:
                        originalCommunity = community
                        communities[originalCommunity].remove(currentVertex)
                        break
                previousCosineSimilarity = []
                for x in communities[originalCommunity]:
                    previousCosineSimilarity.append(cosineSimilarityMatrix[x][currentVertex])
                previousCosineSimilarity = mean(previousCosineSimilarity)
                communities[originalCommunity].append(currentVertex)
                for community in communities.keys():
                    if community == originalCommunity:
                        continue
                    cosine = []
                    for x in communities[community]:
                        cosine.append(cosineSimilarityMatrix[x][currentVertex])
                    QNewman = calculateQNewman(communities[community], currentVertex)
                    QAttribute = mean(cosine)
                    Q = alpha * QNewman + (1 - alpha) *  (QAttribute - previousCosineSimilarity)
                    if Q > 0 and Q > maxCompositeModularityGain:
                        maxCompositeModularityGain = Q
                        communityWithMaxGain = community
                if communityWithMaxGain != -1:
                    convergence = False
                    communities[communityWithMaxGain].append(currentVertex)
                    communities[originalCommunity].remove(currentVertex)
                    if not len(communities[originalCommunity]):
                        del communities[originalCommunity]
            print '\t\t', len(communities)
            if convergence:
                break
    return communities


def communityToNode(communities):
    communityIndex = 0
    membership = list(range(0, g.vcount()))
    for community in communities.keys():
        for vertex in communities[community]:
            membership[vertex] = communityIndex
        communityIndex += 1
    g.contract_vertices(membership, combine_attrs=mean)
    g.simplify(combine_edges=sum)


def mergeCommunities(previousCommunities, communities, iteration):
    if iteration == 0:
        communityIndex = 0
        for community in communities.keys():
            previousCommunities[communityIndex] = communities[community]
            communityIndex += 1
    else:
        newCommunities = dict()
        communityIndex = 0
        for community in communities.keys():
            temp = []
            for vertex in communities[community]:
                temp.extend(previousCommunities[vertex])
            newCommunities[communityIndex] = temp
            communityIndex += 1
        previousCommunities = newCommunities.copy()
    return previousCommunities


def writeCommunities(communities, alpha):
    if alpha == 0.5:
        alpha = 5
    fileName = "communities_" + str(int(alpha)) + ".txt"
    with open(fileName, 'w') as f:
        for community in communities.keys():
            for vertex in sorted(communities[community]):
                f.write(str(vertex) + ", ")
            f.write("\n")


def main():
    if len(sys.argv) <= 1:
        print "python sac1.py <alpha> \n Alpha can be 0, 0.5, or 1."
        exit()
    alpha = float(sys.argv[1])
    readUsers()
    readEdges()
    communities = dict()
    previousCommunities = dict()
    for iteration in range(0, 15):
        print "Main iteration: ", iteration + 1
        if iteration != 0:
            communityToNode(communities)
        communities = sac1(alpha)
        if len(communities) == len(previousCommunities):
            break
        else:
            previousCommunities = mergeCommunities(previousCommunities, communities, iteration)
    writeCommunities(previousCommunities, alpha)


if __name__ == '__main__':
    main()
