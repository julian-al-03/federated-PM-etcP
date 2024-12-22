import threading
from customTypes.types import OrgId, Act, CandId, AnomTrie, Message, OrgPort, CandidateTimestamp
from socketConn import OrgSocket
from organization import Organization
from utils import serealizeAnomTriePrefix, mergeXesLogs
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from damgard_jurik_utils.damgard_jurik_local.crypto import keygen, PublicKey, EncryptedNumber
from utils import serialize_anom_trie
from pm4py.objects.petri_net.utils import incidence_matrix, consumption_matrix
from random import randint
from pm4py.objects.petri_net.semantics import PetriNetSemantics
from utils import printProgressBar, store_anom_trie, load_anom_trie
from time import time
import random

# The coordinator class takes a list of tuples with (Organization Id, port, path to log)
class Coordinator:
    def __init__(self, coordinatorPort: int, orgsInit: list[tuple[OrgId, int, str]], numTraces: int, activityPartition: dict[OrgId, list[Act]] = None):
        self.mergedPrefixTree = AnomTrie()
        self.mergedPrefixTree.weight = numTraces
        self.mergedPrefixTree.mask = [True] * numTraces
        self.coordinatorPort = coordinatorPort
        self.activityPartition = activityPartition
        self.orgPorts: list[OrgPort] = []
        for org in orgsInit:
            self.orgPorts.append(OrgPort(org[0], org[1]))
        self.numTraces = numTraces

        # dicts for storing masks
        self.maskDict = {}
        for org in orgsInit:
            self.maskDict[org[0]] = {}

        # init damgard-jurik
        publicKey, privateKeyShares, private_key_ring = keygen(n_bits=1024, s=1, threshold=2, n_shares=len(orgsInit))
        self.private_key_ring = private_key_ring
        self.publicKey = publicKey

        # init own socket
        self.socket = OrgSocket(coordinatorPort=coordinatorPort, orgPort=coordinatorPort, orgIdentifier="coordinator", orgPorts=self.orgPorts, handler=self.handler)
        self.socket.start()

        # init organizations
        self.orgs: list[Organization] = []
        for org, privateKeyShare in zip(orgsInit, privateKeyShares):
            self.orgs.append(Organization(identifier=org[0], orgPort=org[1], coordinatorPort=self.coordinatorPort, orgPorts=self.orgPorts, path=org[2], numTraces=numTraces, publicKey=publicKey, privateKeyShare=privateKeyShare, activityPartition=activityPartition))

        # show ground truth prefix tree
        # logs = []
        # for org in orgsInit:
        #     logs.append(pm4py.read_xes(org[2]))
        # mergedLog = mergeXesLogs(logs)
        # pm4py.view_prefix_tree(pm4py.discover_prefix_tree(mergedLog))
    
    def storeMergedTree(self, path: str):
        lists = self.getDeanymLists()
        store_anom_trie(self.mergedPrefixTree, path, lists)

    def loadMergedTree(self, path: str):
        tree, lists = load_anom_trie(path)
        self.mergedPrefixTree = tree
        for org in self.orgs:
            org.deanymMap = lists[org.identifier]

    def close(self):

        for org in self.orgs:
            org.socket.close()
        self.socket.close()

    def handler(self, sender: str, message: Message):
        # print(f"Coordinator received message from {sender}: {message}")
        pass

    def syncMergedTrie(self):
        for org in self.orgPorts:
            self.socket.sendToOrgWaitForResponse(org.orgId, Message(action="syncMergedTrie", data=serialize_anom_trie(self.mergedPrefixTree)))

    def startTimer(self):
        return time()

    def stopTimer(self, start: float):
        return time() - start

    def mergeTrees(self, naive: bool = False, oneIteration: bool = False):
        stack = []
        stack.append(self.mergedPrefixTree)
        countNodesGenerated = 0
        iterations = 0
        totalTime = 0

        while stack:
            node = stack.pop()
            children = None
            averageTimePerIteration = 0
            timer = self.startTimer()
            if naive:
                children = self.extendNodeNaive(node)
            else:
                children = self.extendNode(node)
            timeElapsed = self.stopTimer(timer)
            # drop mask of node after extending
            # not needed anymore and would only take up memory
            node.mask = None

            stack += children
            countNodesGenerated += len(children)
            print(f"Generated {countNodesGenerated} nodes")

            iterations += 1
            totalTime += timeElapsed
            averageTimePerIteration = totalTime/iterations
            print(f"Avg time per iteration: {averageTimePerIteration}")

            # process = psutil.Process()
            # print("Memory Usage:" + str(process.memory_info().rss))

            if oneIteration:
                return self.mergedPrefixTree
        self.syncMergedTrie()
        return self.mergedPrefixTree
    
    def deanymAnomTrie(self, root: AnomTrie, fullDeanym: bool = False, unsafeNodes: dict[CandId, list[Act]] = None) -> AnomTrie:

        # traverse the tree
        stack = [root]
        while stack:
            # if eventuallyReachWeightOne is true, do not deanymize
            node = stack.pop()
            if node.candId is not None:
                unsafeActs = []
                if not fullDeanym:
                    if node.candId in unsafeNodes:
                        unsafeActs = unsafeNodes[node.candId]
                res = self.socket.sendToOrgWaitForResponse(node.candId.orgId, Message(action="deanymAct", data={"candId": node.candId, "unsafeActs": unsafeActs}))
                if res.data:
                    node.label = res.data
                    node.act = res.data
            stack += node.children

        return root
    

    def addsensitiveActivitiesToTrie(self, sensitiveActivitiesLists: list[tuple[OrgId, list[Act]]]):

        def getPrefix(node: AnomTrie):
            prefix = []
            current = node
            while current is not None:
                if current.candId is not None:
                    prefix = [current.candId] + prefix
                current = current.parent
            return prefix

        # traverse all leave nodes
        stack: list[AnomTrie] = [self.mergedPrefixTree]
        while stack:
            node = stack.pop()
            if not node.children:
                sensitiveActivitiesMap = {}
                for orgId, sensitiveActivitiesList in sensitiveActivitiesLists:
                    for act in sensitiveActivitiesList:
                        res = self.socket.sendToOrgWaitForResponse(orgId, Message(action="hasSensitiveActivity", data={"act": act, "candIds": getPrefix(node)}))
                        sensitiveActivitiesMap[(orgId,act)] = res.data
                node.sensitiveActivities = sensitiveActivitiesMap
            stack += node.children

    def calculateUnsafeNodes(self, sensitiveActivitiesLists: list[tuple[OrgId, list[Act]]]) -> dict[CandId, list[Act]]:
        self.addsensitiveActivitiesToTrie(sensitiveActivitiesLists)
        self.syncMergedTrie()
        unsafeNodes: dict[CandId, list[Act]] = {}
        for org in self.orgPorts:
            res = self.socket.sendToOrgWaitForResponse(org.orgId, Message(action="calculateUnsafeNodes", data=sensitiveActivitiesLists))
            res = res.data
            for candId, acts in res.items():
                if candId not in unsafeNodes:
                    unsafeNodes[candId] = acts
                unsafeNodes[candId] += acts
        return unsafeNodes


    def extendNode(self, node: AnomTrie) -> list[AnomTrie]:
        # get candidates
        candidates: list[CandId] = self.getCandidates(node)
        if len(candidates) == 0:
            return []
        # get unmasked comparison result
        response = self.socket.sendToOrgWaitForResponse(candidates[0].orgId, Message(action="comparison", data={"candList": candidates, "currentRes": None, "mask": self.getMask(node)}))
        weightList, maskList = response.data
        children = []
        for candId, weight, mask in zip(candidates, weightList, maskList):
            if weight > 0:
                newChild = AnomTrie()
                newChild.candId = candId
                newChild.label = str(candId.orgId) + "_" + str(candId.id)
                newChild.mask = mask
                newChild.weight = weight
                newChild.weightFrac = newChild.weight/candId.weight
                newChild.parent = node
                children.append(newChild)
        node.children = children
        return children
        
    def extendNodeNaive(self, node: AnomTrie) -> list[AnomTrie]:
        # get candidates
        candidates = self.getCandidates(node)
        if len(candidates) == 0:
            return []
        # get unmasked comparison result
        response = self.socket.sendToOrgWaitForResponse(candidates[0].orgId, Message(action="naiveCompare", data={"candList": candidates, "currentRes": None, "currentPosition": 0}))
        unmaskedComparison = response.data
        # apply the mask of the current node
        comparison = self.applyMaskNaive(unmaskedComparison, node)

        # create children
        children = []
        for cand in candidates:
            mask = [False] * self.numTraces
            atLeastOne = False
            weight = 0
            for i in range(self.numTraces):
                if comparison[i] == cand:
                    mask[i] = True
                    weight += 1
                    atLeastOne = True
            if atLeastOne:
                newChild = AnomTrie()
                newChild.candId = cand
                newChild.weight = weight
                newChild.weightFrac = newChild.weight/cand.weight
                newChild.label = str(cand.orgId) + " " + str(cand.id) + " " + str(newChild.weight/cand.weight)
                newChild.mask = mask
                newChild.parent = node
                children.append(newChild)
                # for debugging
                self.deanymAnomTrie(newChild, fullDeanym=True)
        node.children = children
        return children

    def applyMaskNaive(self, comparison: list[CandId | None], node: AnomTrie):
        mask = node.mask
        for i in range(self.numTraces):
            if mask[i] is False:
                comparison[i] = False
        return comparison

    def getCandidates(self, node: AnomTrie):
        candidates = []
        for org in self.orgPorts:
            prefix = serealizeAnomTriePrefix(node)
            response: Message = self.socket.sendToOrgWaitForResponse(org.orgId, Message(action="generateCandidates", data={"prefix": prefix}))
            candidates += response.data
        return candidates
    
    def setMask(self, node: AnomTrie, mask: list[bool]):
        node.mask = mask
    
    def getMask(self, node: AnomTrie):
        if node.candId is None:
            res = [None] * self.numTraces
            encNum = self.publicKey.encrypt(1)
            for i in range(self.numTraces):
                res[i] = encNum.value
            return res
        return node.mask
    
    def getDeanymLists(self) -> dict[OrgId, dict[int, Act]]:
        # iterate through all organizations and get the deanymization lists
        deanymLists = {}
        for org in self.orgPorts:
            response = self.socket.sendToOrgWaitForResponse(org.orgId, Message(action="getDeanymList", data=None))
            deanymLists[org.orgId] = response.data
        return deanymLists
    
    def trieToCandPrefixes(self) -> tuple[list[list[CandId]], list[int]]:
        stack = [self.mergedPrefixTree]
        prefixes = []
        prefixWeights = []

        def nodeToPrefix(node: AnomTrie):
            prefix = []
            current = node
            while current is not None:
                if(current.candId is not None):
                    prefix = [current.candId] + prefix
                current = current.parent
            return prefix
        
        while stack:
            current = stack.pop()
            if sum([child.weight for child in current.children]) < current.weight:
                prefixes.append(nodeToPrefix(current))
                prefixWeights.append(current.weight - sum([child.weight for child in current.children]))
            stack += current.children
        
        return prefixes, prefixWeights

    def approximateAlignments(self, modelLanguage: list[list[Act]]):
        candPrefixes, _ = self.trieToCandPrefixes()
        res = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="approximateAlignments", data={"candPrefixes": candPrefixes, "modelLanguage": modelLanguage}))
        return res.data
    
    def replayLog(self, model: PetriNet, initialMarking: Marking, finalMarking: Marking, preTransitions: list[str] = None, postTransitions: list[str] = None, sampleSize: float = 1.0, oneIteration: bool = False) -> float:

        incidenceMatrix = incidence_matrix.construct(model)
        consumptionMatrix = consumption_matrix.construct(model)

        def transitionNameToIndex(transitionName: str) -> int:
            for key, transition in incidenceMatrix.transitions.items():
                if key.name == transitionName:
                    return transition

        # send consumption matrix to organizations
        for org in self.orgPorts:
            self.socket.sendToOrgWaitForResponse(org.orgId, Message(action="setMatrices", data=(incidenceMatrix, consumptionMatrix)))

        # get candidate prefixes
        candPrefixes, prefixWeights = self.trieToCandPrefixes()
        # sample the traces
        if sampleSize < 1.0:
            sampleSize = int(len(candPrefixes) * sampleSize)
            # random.seed(0)
            candPrefixes = random.sample(candPrefixes, sampleSize)
            # random.seed(0)
            prefixWeights = random.sample(prefixWeights, sampleSize)

        # print info
        avgLength = sum([len(prefix) for prefix in candPrefixes])/len(candPrefixes)
        print(f"Longest trace: {max([len(prefix) for prefix in candPrefixes])}")
        print(f"Avergage trace length: {avgLength}")
        
        # if oneIteration is true, only replay one trace
        # find a trace with average length
        # remove all other traces
        print("oneIteration", oneIteration)
        if oneIteration:
            for i, prefix in enumerate(candPrefixes):
                if len(prefix) == int(avgLength):
                    candPrefixes = [prefix]
                    prefixWeights = [prefixWeights[i]]
                    break
        print(f"Replaying {len(candPrefixes)} traces")
        # fire preTransitions
        initialMarkingVec = incidenceMatrix.encode_marking(initialMarking)
        for transition in preTransitions:
            columnIndex = transitionNameToIndex(transition)
            for i in range(len(incidenceMatrix.places)):
                initialMarkingVec[i] += incidenceMatrix.a_matrix[i][columnIndex]
        for i, place in enumerate(initialMarkingVec):
            initialMarkingVec[i] = self.publicKey.encrypt(place)
        
        # replay traces
        totalEscapingEdges = self.publicKey.encrypt(0)
        totalAllowedTransitions = self.publicKey.encrypt(0)
        count = 0
        printProgressBar(count, len(candPrefixes), prefix = 'Replaying Traces:', suffix = 'Complete')
        test = sum(prefixWeights)
        notFittingCount = 0
        for prefix, weight in zip(candPrefixes, prefixWeights):
            numEscapingEdges, numAllowedTransitions, finalMarkingVec, fitting = self.replayTrace(trace=prefix, incidenceMatrix=incidenceMatrix, consumptionMatrix=consumptionMatrix, currentMarkingVec=initialMarkingVec)

            # multiply fitting with numEscapingEdges and numAllowedTransitions
            temp = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": fitting.value, "num2": numEscapingEdges.value}))
            numEscapingEdges = EncryptedNumber(temp.data, self.publicKey)
            temp = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": fitting.value, "num2": numAllowedTransitions.value}))
            numAllowedTransitions = EncryptedNumber(temp.data, self.publicKey)

            # fire postTransitions
            for postTransition in postTransitions:
                columnIndex = transitionNameToIndex(postTransition)
                for i in range(len(incidenceMatrix.places)):
                    finalMarkingVec[i] += self.publicKey.encrypt(incidenceMatrix.a_matrix[i][columnIndex])

            # check if final marking is reached
            traceIsFitting = self.checkIfFinalMarking(finalMarkingVec, incidenceMatrix.encode_marking(finalMarking))

            # multiply traceIsFitting with numEscapingEdges and numAllowedTransitions
            temp = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": traceIsFitting.value, "num2": numEscapingEdges.value}))
            numEscapingEdges = EncryptedNumber(temp.data, self.publicKey)
            temp = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": traceIsFitting.value, "num2": numAllowedTransitions.value}))
            numAllowedTransitions = EncryptedNumber(temp.data, self.publicKey)
            
            # add to total        
            totalEscapingEdges += numEscapingEdges * weight
            totalAllowedTransitions += numAllowedTransitions * weight
    
            count += 1
            printProgressBar(count, len(candPrefixes), prefix = 'Replaying Traces:', suffix = 'Complete')
        print(f"Number of traces not fitting: {notFittingCount}")
        precisionValue = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="calculatePrecision", data={"totalEscapingEdges": totalEscapingEdges.value, "totalAllowedTransitions": totalAllowedTransitions.value}))
        return precisionValue.data


    def replayTrace(self, trace: list[CandId], incidenceMatrix: incidence_matrix.IncidenceMatrix, consumptionMatrix: consumption_matrix.ConsumptionMatrix, currentMarkingVec: list[int]) -> tuple[EncryptedNumber, EncryptedNumber, list[int]]:
        
        def fire(marking: list[int], candId: CandId):
            marking1 = [num.value for num in marking]
            res = self.socket.sendToOrgWaitForResponse(candId.orgId, Message(action="fire", data={"candId": candId, "marking": marking1}))
            markingRes, isEnabled = res.data
            return [EncryptedNumber(num, self.publicKey) for num in markingRes], isEnabled
        
        def getEnabledTransitions(marking: list[int]) -> list[int]:
            res = [None] * len(consumptionMatrix.transitions)
            for transition, transitionIndex in consumptionMatrix.transitions.items():
                if transition.label is None:
                    res[consumptionMatrix.transitions[transition]] = None
                    continue
                places = []
                for i in range(len(consumptionMatrix.places)):
                    if consumptionMatrix.c_matrix[i][transitionIndex] == -1:
                        places.append(marking[i].value)
                # todo
                # https://www.wolframalpha.com/input?i2d=true&i=%5C%2840%29a%2Bb%5C%2841%29*%5C%2840%29c%2Bd%5C%2841%29*%5C%2840%29e%2Bf%5C%2841%29
                # add random values to all values in enabled
                # for i in range(len(enabled)):
                #     r = randint(0, 100)
                #     c = EncryptedNumber(enabled, self.publicKey) + r
                #     enabled[i] = c.value
                if len(places) == 0:
                    res[consumptionMatrix.transitions[transition]] = self.publicKey.encrypt(1)
                else:
                    enabled = (self.publicKey.encrypt(1)).value
                    for place in places:
                        temp = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": enabled, "num2": place}))
                        enabled = temp.data
                    res[consumptionMatrix.transitions[transition]] = enabled                     
            return res

        currentNode = self.mergedPrefixTree
        numEscapingEdges = self.publicKey.encrypt(0)
        numAllowedTransitions = self.publicKey.encrypt(0)
        isEnabledProd = self.publicKey.encrypt(1)
        for i, candId in enumerate([None] + trace):
            # fire
            if candId is not None:
                currentMarkingVec, isEnabled = fire(currentMarkingVec, candId)
                multEnabled = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": isEnabledProd.value, "num2": isEnabled}))
                multEnabled = multEnabled.data
                isEnabledProd = EncryptedNumber(multEnabled, self.publicKey)
                # get the next node
                for child in currentNode.children:
                    if child.candId == candId:
                        currentNode = child
                        break
            # count allowed transitions
            allowedTransitions = getEnabledTransitions(currentMarkingVec)
            # test = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="decryptListDebug", data=allowedTransitions))
            for allowedTransition in allowedTransitions:
                if allowedTransition is not None:
                    numAllowedTransitions += EncryptedNumber(allowedTransition, self.publicKey)
            # remove reflected transitions from allowedTransitions
            for child in currentNode.children:
                res = self.socket.sendToOrgWaitForResponse(child.candId.orgId, Message(action="isEscapingEdge", data={"candId": child.candId, "allowedTransitions": allowedTransitions}))
                allowedTransitions = res.data
            # test = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="decryptListDebug", data=allowedTransitions))
            # count escaping edges (allowed transitions - reflected transitions)
            for allowedTransition in allowedTransitions:
                if allowedTransition is not None:
                    numEscapingEdges += EncryptedNumber(allowedTransition, self.publicKey)

        return numEscapingEdges, numAllowedTransitions, currentMarkingVec, isEnabledProd

        
    def checkIfFinalMarking(self, encMarking: list[EncryptedNumber], finalMarking: list[int]) -> EncryptedNumber:
        total = (self.publicKey.encrypt(1)).value
        for i in range(len(encMarking)):
            diff = encMarking[i] - self.publicKey.encrypt(finalMarking[i])
            # mult diff*diff
            innerSquare = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": diff.value, "num2": diff.value}))
            innerSquare = EncryptedNumber(innerSquare.data, self.publicKey)
            # subtract 1 from innerSquare
            innerSquare -= self.publicKey.encrypt(1)
            outerSquare = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": innerSquare.value, "num2": innerSquare.value}))
            outerSquare = outerSquare.data
            # mult total*outerSquare
            total = self.socket.sendToOrgWaitForResponse(self.orgPorts[0].orgId, Message(action="multNumbers", data={"num1": total, "num2": outerSquare}))
            total = total.data
        return EncryptedNumber(total, self.publicKey)
    