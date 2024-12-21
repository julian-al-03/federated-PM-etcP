# a class that takes an identifier (string) and a port number (int) and a path to a file (string) as inputs and creates a new organization object
# the class also has a private function deanym that takes CanId as input and returns a Act

from random import randint
import random
from typing import List, Tuple
from socketConn import OrgSocket
from pm4py.objects.log.obj import EventLog
import pm4py
from customTypes.types import OrgId, Act, CandId, Message, OrgTrie, CandidateTimestamp, ActivityTimestamp, EncCandId, PartialEncCandId, PartialEncTime, OrgPort, AnomTrie
import time
from damgard_jurik_utils.damgard_jurik_local.crypto import PrivateKeyShare, PublicKey, EncryptedNumber
import concurrent.futures
import os
from pm4py.objects.log.obj import Trace
from utils import printProgressBar, deserialize_anom_trie
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.utils import incidence_matrix, consumption_matrix
from functools import partial

class Organization:
    def __init__(self, identifier: OrgId, orgPort: int, coordinatorPort: int, orgPorts: list[OrgPort], path: str, numTraces: int, publicKey: PublicKey, privateKeyShare: PrivateKeyShare, activityPartition: dict[OrgId, list[Act]] = None):
        self.identifier = identifier
        self.orgPorts = orgPorts
        self.coordinatorPort = coordinatorPort
        self.path = path
        self.numTraces = numTraces
        self.privateKeyShare = privateKeyShare
        self.publicKey = publicKey
        self.deanymMap: dict[int, Act] = {}
        self.candTimeMap: dict[CandId, list[CandidateTimestamp]] = {}
        self.nextCandidateId = 0
        self.privateKeyShare = privateKeyShare
        if path is not None:
            self.log = pm4py.read_xes(path)
            self.prefixTree: OrgTrie = self.discoverPrefixTree()
        self.socket = OrgSocket(orgIdentifier=identifier, orgPort=orgPort, coordinatorPort=coordinatorPort, orgPorts=orgPorts, handler=self.handler)
        self.socket.start()
        self.numCores = os.cpu_count()
        self.activityPartition = activityPartition
        self.incidenceMatrix: incidence_matrix.IncidenceMatrix = None
        self.consumptionMatrix: consumption_matrix.ConsumptionMatrix = None
        self.mergedTree: AnomTrie = None

        self.activityToOrgId = {}
        if activityPartition is not None:
            for orgId, acts in activityPartition.items():
                for act in acts:
                    self.activityToOrgId[act] = orgId

        # orgToInt and IntToOrg
        self.orgToInt = {}
        self.intToOrg = {}
        for i, org in enumerate(orgPorts):
            self.orgToInt[org.orgId] = i
            self.intToOrg[i] = org.orgId


    def handler(self, sender: str, message: Message):
        # print(f"{self.identifier} received message from {message.sender}: {message}")
        if(message.action == "generateCandidates"):
            prefix = message.data["prefix"]
            candidates = self.generateCandidates(prefix)
            self.socket.sendToCoordinator(Message(action="candidates", data=candidates, id=message.id))

        elif(message.action == "naiveCompare"):
            candList = message.data["candList"]
            currentRes = message.data["currentRes"]
            currentPosition = message.data["currentPosition"]
            res = self.naiveComparison(candList, currentRes, currentPosition)
            if message.sender == "coordinator":
                self.socket.sendToCoordinator(Message(action="naiveComparisonResult", data=res, id=message.id))
            else:
                self.socket.sendToOrg(message.sender, Message(action="naiveComparisonResult", data=res, id=message.id))

        elif(message.action == "comparison"):
            candList = message.data["candList"]
            currentRes = message.data["currentRes"]
            mask = message.data["mask"]
            res = self.comparison(candList, currentRes, mask)
            if message.sender == "coordinator":
                self.socket.sendToCoordinator(Message(data=res, id=message.id))
            else:
                self.socket.sendToOrg(message.sender, Message(data=res, id=message.id))

        elif(message.action == "decryptAndCompare"):
            timestamps1 = message.data["timestamps1"]
            timestamps2 = message.data["timestamps2"]
            finalComparison = message.data["finalComparison"]
            candList = message.data["candList"]
            mask = message.data["mask"]
            res = self.decryptAndCompare(timestamps1, timestamps2, candList, finalComparison, mask)
            self.socket.sendToOrg(message.sender, Message(data=res, id=message.id))

        elif(message.action == "decryptList"):
            list = message.data["list"]
            c_list = message.data["c_list"]
            i = message.data["i"]
            res = self.decryptList(list, c_list, i)
            self.socket.sendToOrg(message.sender, Message(data=res, id=message.id))

        elif(message.action == "smallerZero"):
            list = message.data["list"]
            c_list = message.data["c_list"]
            i = message.data["i"]
            res = self.smallerZero(list, c_list, i)
            self.socket.sendToOrg(message.sender, Message(data=res, id=message.id))

        elif(message.action == "deanymAct"):
            # deanym activity given
            res = self.deanym([message.data["candId"]])
            if res[0] in message.data["unsafeActs"]:
                self.socket.sendToCoordinator(Message(data=False, id=message.id))
            else: 
                self.socket.sendToCoordinator(Message(data=res[0], id=message.id))
        
        elif(message.action == "getDeanymList"):
            # deanym activity given
            res = self.deanymMap
            self.socket.sendToCoordinator(Message(data=res, id=message.id))

        elif(message.action == "approximateAlignments"):
            modelLanguage = message.data["modelLanguage"]
            candPrefixes = message.data["candPrefixes"]
            res = self.approximateAlignments(modelLanguage, candPrefixes)
            self.socket.sendToCoordinator(Message(data=res, id=message.id))

        elif(message.action == "setMatrices"):
            incMat, conMat = message.data
            self.incidenceMatrix = incMat
            self.consumptionMatrix = conMat
            self.socket.sendToCoordinator(Message(data=True, id=message.id))

        elif(message.action == "fire"):
            candId = message.data["candId"]
            marking = message.data["marking"]
            marking, isEnabled = self.fire(candId, marking)
            self.socket.sendToCoordinator(Message(data=(marking, isEnabled), id=message.id))

        elif(message.action == "multNumbers"):
            num1 = message.data["num1"]
            num2 = message.data["num2"]
            res = self.multNumbers(num1, num2)
            if message.sender == "coordinator":
                self.socket.sendToCoordinator(Message(data=res, id=message.id))
            else:
                self.socket.sendToOrg(message.sender, Message(data=res, id=message.id))

        elif(message.action == "multiplyEncryptedList"):
            list = message.data
            res = self.multiplyEncryptedList(list)
            if message.sender == "coordinator":
                self.socket.sendToCoordinator(Message(data=res, id=message.id))
            else:
                self.socket.sendToOrg(message.sender, Message(data=res, id=message.id))

        elif(message.action == "isEscapingEdge"):
            candId = message.data["candId"]
            allowedTransitions = message.data["allowedTransitions"]
            res = self.isEscapingEdge(candId, allowedTransitions)
            self.socket.sendToCoordinator(Message(data=res, id=message.id))

        elif(message.action == "isZero"):
            list = message.data
            res = self.isZero(list)
            self.socket.sendToCoordinator(Message(data=res, id=message.id))

        elif(message.action == "calculatePrecision"):
            totalEscapingEdges = message.data["totalEscapingEdges"]
            totalAllowedTransitions = message.data["totalAllowedTransitions"]
            res = self.calculatePrecision(totalEscapingEdges, totalAllowedTransitions)
            self.socket.sendToCoordinator(Message(data=res, id=message.id))
        
        elif(message.action == "decryptListDebug"):
            list = message.data
            res = self.decryptListDebug(list)
            if message.sender == "coordinator":
                self.socket.sendToCoordinator(Message(data=res, id=message.id))
            else:
                self.socket.sendToOrg(message.sender, Message(data=res, id=message.id))

        elif(message.action == "syncMergedTrie"):
            self.mergedTree = deserialize_anom_trie(message.data)
            self.socket.sendToCoordinator(Message(data=True, id=message.id))

        elif(message.action == "hasSensitiveActivity"):
            act = message.data["act"]
            candIds = message.data["candIds"]
            res = self.hasSensitiveActivity(act, candIds)
            self.socket.sendToCoordinator(Message(data=res, id=message.id))

        elif(message.action == "confidenceUnderTreshold"):
            confidences = message.data["confidences"]
            c_list = message.data["c_list"]
            i = message.data["i"]
            res = self.confidenceUnderTreshold(confidences, c_list, i)
            self.socket.sendToOrg(message.sender, Message(data=res, id=message.id))

        elif(message.action == "calculateUnsafeNodes"):
            sensitiveActivitiesLists = message.data
            res = self.calculateUnsafeNodes(sensitiveActivitiesLists)
            self.socket.sendToCoordinator(Message(data=res, id=message.id))

        elif(message.action == "close"):
            if self.socket.handle_peer_connection_thread is not None:
                self.socket.handle_peer_connection_thread.join()
            if self.socket.listener_thread is not None:
                self.socket.listener_thread.join()



    def decryptListDebug(self, list: list[int]):
        enc_list = []
        c_list = []
        for enc in list:
            if enc is None:
                continue
            enc_list.append(enc)
            c_list.append(self.privateKeyShare.decrypt(EncryptedNumber(enc, self.publicKey)))
        otherOrg = None
        for org in self.orgPorts:
            if org.orgId != self.identifier:
                otherOrg = org
                break
        res = self.socket.sendToOrgWaitForResponse(otherOrg.orgId, Message(action="decryptList", data={
            "list": enc_list,
            "c_list": c_list,
            "i": self.privateKeyShare.i
        }))
        return res.data

    def hasSensitiveActivity(self, sensitiveAct: Act, candIds: list[CandId]):
        # deanym candId
        prefix = self.deanym(candIds)
        # if act is in prefix return encrypted 1 else 0
        for act in prefix:
            if act == sensitiveAct:
                return (self.publicKey.encrypt(1)).value
        return (self.publicKey.encrypt(0)).value

    def deanym(self, candIds: list[CandId]) -> list[Act]:
        res = []
        for candId in candIds:
            if candId.orgId == self.identifier:
                res.append(self.deanymMap[candId.id])
            else:
                res.append(candId)
        return res
    
    def createCandidate(self, activity: Act):
        candId = CandId(self.identifier, self.nextCandidateId)
        self.deanymMap[self.nextCandidateId] = activity
        self.nextCandidateId += 1
        return candId
    
    def reduceToOwnPrefix(self, prefix: list[CandId]):
        res = []
        for candId in prefix:
            if candId.orgId == self.identifier:
                res.append(candId)
        return res
            
    def nextActivities(self, prefix: list[Act]) -> list[OrgTrie]:
        node = self.prefixTree
        for act in prefix:
            for child in node.children:
                if child.label == act:
                    node = child
                    break
        return node.children

    def generateCandidates(self, CandId):
        reducedPrefix = self.reduceToOwnPrefix(CandId)
        prefix = self.deanym(reducedPrefix)
        nextActs = self.nextActivities(prefix)
        res = []
        for act in nextActs:
            newCandPrefix = prefix + [act.act]
            newCand = self.createCandidate(act.act)
            prefixNode = self.prefixToNode(newCandPrefix)
            newCand.weight = prefixNode.weight
            # convert activity timestamps to candidate timestamps
            activityTimestamps = prefixNode.timestamps
            candidateTimestamps = []
            for timestamp in activityTimestamps:
                if timestamp is None:
                    candidateTimestamps.append(None)
                else:
                    candidateTimestamps.append(CandidateTimestamp(newCand, timestamp.time))
            self.candTimeMap[newCand.id] = candidateTimestamps
            res.append(newCand)
        return res
    
    def prefixToNode(self, prefix: list[Act]):
        node = self.prefixTree
        for act in prefix:
            for child in node.children:
                if child.label == act:
                    node = child
                    break
        return node
    
    def getTimestamps(self, candId: CandId):
        # link to timestmaps from canTimeMap are removed after access
        if candId is None:
            return [0] * self.numTraces
        res = self.candTimeMap[candId.id]
        del self.candTimeMap[candId.id]
        return res
    
    def naiveComparison(self, candList: list[CandId], currentRes: list[CandidateTimestamp] | None, currentPosition = 0):
        firstExecution = False
        # find own candidate:
        if currentPosition == 0:
            firstExecution = True
            currentRes = self.getTimestamps(candList[currentPosition])
            currentPosition += 1
        else:
            ownTimestamps = self.getTimestamps(candList[currentPosition])
            for i in range(self.numTraces):
                if ownTimestamps[i] is None:
                    continue
                elif currentRes[i] is None:
                    currentRes[i] = ownTimestamps[i]
                elif(ownTimestamps[i].time < currentRes[i].time):
                    currentRes[i] = ownTimestamps[i]
            currentPosition +=1
        
        # pass on for the next comparison
        res = None
        if currentPosition == len(candList):
            res = currentRes
        elif candList[currentPosition].orgId == self.identifier:
            res = self.naiveComparison(candList, currentRes, currentPosition)
        else:
            response = self.socket.sendToOrgWaitForResponse(candList[currentPosition].orgId, Message(action="naiveCompare", data={
                "candList": candList,
                "currentRes": currentRes,
                "currentPosition": currentPosition
            }))
            res = response.data
        
        # pass to coordinator (reduced) or next org (non reduced)
        if(firstExecution):
            res = self.reduceListToCandIdList(res)
        return res
    
    def comparison(self, candList: list[CandId], currentRes: list[CandidateTimestamp] | None, mask: list[int]):
        # get list of invloved organizations (no duplicates)
        seen = set()
        orgList = [candId.orgId for candId in candList if not (candId.orgId in seen or seen.add(candId.orgId))]
        nextCandidate = orgList.index(self.identifier) + 1
        ownIndex = orgList.index(self.identifier)
        previousCandidate = orgList.index(self.identifier) - 1

        # get own timestamps
        ownCandIds = [cand for cand in candList if cand.orgId == self.identifier]
        ownTimestamps = self.compressCandidates(ownCandIds)
        start = time.time()
        # print('encrypting timestamps')
        encTimestamps = self.encryptCandidateTimestamps(ownTimestamps)
        # print('timestamps encrypted')
        # print(time.time() - start)

        # variables for the final result
        weightList: list[int]
        maskList: list[list[int]]

        # special case if all the candidates are from the same org
        if len(orgList) == 1:
            # init result variables
            weightList = [None] * len(candList)
            for i in range(len(weightList)):
                weightList[i] = self.publicKey.encrypt(0)
            # transform mask to list of encryption number objects
            mask = [EncryptedNumber(mask[i], self.publicKey) for i in range(self.numTraces)]
            maskList = [None] * len(candList)
            for i in range(len(maskList)):
                maskList[i] = [None] * self.numTraces
            for i, timestamp in enumerate(ownTimestamps):
                for j, cand in enumerate(candList):
                    if timestamp is not None and cand.id == timestamp.candId.id:
                        encNumber = mask[i]
                        maskList[j][i] = encNumber.value
                        weightList[j] = weightList[j] + encNumber
                    else:
                        encNumber = self.publicKey.encrypt(0)
                        maskList[j][i] = encNumber.value
            # decrypt weight list (with help of another org)
            c_list = [None] * len(weightList)
            for i, weight in enumerate(weightList):
                c_list[i] = self.privateKeyShare.decrypt(weight)
            otherOrg = None
            for org in self.orgPorts:
                if org.orgId != self.identifier:
                    otherOrg = org
                    break
            res = self.socket.sendToOrgWaitForResponse(otherOrg.orgId, Message(action="decryptList", data={
                "list": weightList,
                "c_list": c_list,
                "i": self.privateKeyShare.i
            }))
            weightList = res.data

            return weightList, maskList


        # garble and partial-decrypt timestamps
        if currentRes is None:
            currentRes = encTimestamps
        else:
            timestamps1, timestamps2, randomList, permutation = self.garbleTimestamps(currentRes, encTimestamps)
            finalComparison = False
            if (nextCandidate == len(orgList)):
                finalComparison = True
            timestamps1 = self.partialDecryptCandidateTimestamps(timestamps1, decryptTime=True, decryptCand=finalComparison)
            timestamps2 = self.partialDecryptCandidateTimestamps(timestamps2, decryptTime=True, decryptCand=finalComparison)
            if finalComparison:
                mask = self.applyPermutation(mask, permutation)
            res: list[CandidateTimestamp] | tuple[list[int], list[int]] = self.socket.sendToOrgWaitForResponse(orgList[previousCandidate], Message(action="decryptAndCompare", data={
                "timestamps1": timestamps1,
                "timestamps2": timestamps2,
                "finalComparison": finalComparison,
                "candList": candList,
                "mask": mask
            }))
            res = res.data
            if finalComparison:
                weightList = res[0]
                maskList = res[1]
                for i in range(len(maskList)):
                    maskList[i] = self.applyPermutation(maskList[i], self.inversePermutation(permutation))
            else:
                res = self.addEncryptedListToCandTime(res, self.multiplyMinusOne(randomList))
                res = self.applyPermutation(res, self.inversePermutation(permutation))
                currentRes = res

        # pass on for the next comparison (if not last in list)
        if (not nextCandidate == len(orgList)):
            res = self.socket.sendToOrgWaitForResponse(orgList[nextCandidate], Message(action="comparison", data={
                "candList": candList,
                "currentRes": currentRes,
                "mask": mask
            }))
            weightList, maskList = res.data
        if ownIndex == 0:
            # decrypt weight list
            c_list = [None] * len(weightList)
            for i, weight in enumerate(weightList):
                c_list[i] = self.privateKeyShare.decrypt(weight)
            res = self.socket.sendToOrgWaitForResponse(orgList[nextCandidate], Message(action="decryptList", data={
                "list": weightList,
                "c_list": c_list,
                "i": self.privateKeyShare.i
            }))
            weightList = res.data
        return weightList, maskList
        


    def garbleTimestamps(self, candList1: list[CandidateTimestamp], candList2: list[CandidateTimestamp]) -> tuple[list[CandidateTimestamp], list[CandidateTimestamp], list[int], list[int]]:
        # permute
        permutation = self.generateRandomPermutation()
        candList1 = self.applyPermutation(candList1, permutation)
        candList2 = self.applyPermutation(candList2, permutation)
        # add random values
        randomList = self.generateRandomEncryptedList((0, 100))
        candList1 = self.addEncryptedListToCandTime(candList1, randomList)
        candList2 = self.addEncryptedListToCandTime(candList2, randomList)

        return candList1, candList2, randomList, permutation


    # function takes two lists with partially decrypted timestamps and compares them
    # afterwards the result is encrypted again and returned
    def decryptAndCompare(self, candList1: list[CandidateTimestamp], candList2: list[CandidateTimestamp], candidateList: list[CandId] = None, finalComparison: bool = False, mask: list[int] = None) -> list[CandidateTimestamp] | tuple[list[int], list[list[int]]]:
        res: list[CandidateTimestamp] = [None] * self.numTraces
        candList1: list[CandidateTimestamp] = self.decryptCandidateTimestamps(candList1, decryptTime=True, decryptCand=finalComparison)
        candList2: list[CandidateTimestamp] = self.decryptCandidateTimestamps(candList2, decryptTime=True, decryptCand=finalComparison)
        weightList = [None] * len(candidateList)
        for i in range(len(weightList)):
            weightList[i] = self.publicKey.encrypt(0)
        maskList = [None] * len(candidateList)
        for i in range(len(maskList)):
            maskList[i] = [None] * self.numTraces
        for i in range(self.numTraces):
            if candList1[i].time == candList2[i].time:
                if finalComparison:
                    encNumber = self.publicKey.encrypt(0)
                    for j, _ in enumerate(candidateList):
                        maskList[j][i] = encNumber.value
                else:
                    # candList1 or candList2, does not matter
                    encNumber = self.publicKey.encrypt(candList2[i].time)
                    res[i] = CandidateTimestamp(candList2[i].candId, encNumber.value)
            elif candList1[i].time < candList2[i].time:
                if finalComparison:
                    for j, cand in enumerate(candidateList):
                        if cand.id == candList1[i].candId.id and cand.orgId == candList1[i].candId.orgId:
                            encNumber: EncryptedNumber = EncryptedNumber(mask[i], self.publicKey)
                            maskList[j][i] = encNumber.value
                            weightList[j] = weightList[j] + encNumber
                        else:
                            encNumber = self.publicKey.encrypt(0)
                            maskList[j][i] = encNumber.value
                else:
                    encNumber = self.publicKey.encrypt(candList1[i].time)
                    res[i] = CandidateTimestamp(candList1[i].candId, encNumber.value)
            else:
                if finalComparison:
                    for j, cand in enumerate(candidateList):
                        if cand.id == candList2[i].candId.id and cand.orgId == candList2[i].candId.orgId:
                            encNumber: EncryptedNumber = EncryptedNumber(mask[i], self.publicKey)
                            maskList[j][i] = encNumber.value
                            weightList[j] = weightList[j] + encNumber
                        else:
                            encNumber = self.publicKey.encrypt(0)
                            maskList[j][i] = encNumber.value
                else:
                    encNumber = self.publicKey.encrypt(candList2[i].time)
                    res[i] = CandidateTimestamp(candList2[i].candId, encNumber.value)
        if finalComparison:
            for i in range(len(weightList)):
                weightList[i] = weightList[i].value
            return weightList, maskList
        else:
            return res
        
    def decryptList(self, list: list[int], c_list: list[int], i: int):
        res = [None] * len(list)
        i_list = [i, self.privateKeyShare.i]
        for i in range(len(list)):
            res[i] = self.privateKeyShare.final_decrypt(
                [c_list[i], self.privateKeyShare.decrypt(list[i])],
                i_list
            )
        return res
            
            
    def compressCandidates(self, candidates: list[CandId]) -> list[CandidateTimestamp]:
        res = [None] * self.numTraces
        for cand in candidates:
            timestamps = self.getTimestamps(cand)
            for i in range(self.numTraces):
                if timestamps[i] is None:
                    continue
                if res[i] is None or timestamps[i].time < res[i].time:
                    res[i] = timestamps[i]
        return res

    def applyFuncToArray(self, func, array: list[any]) -> list[any]:
        # helper function that applies the given function to each element in the list
        res = []
        for i, elem in enumerate(array):
            res.append(func(elem))
        return res
        # parallel version is slower for some reason
        with concurrent.futures.ThreadPoolExecutor() as executor:
            res = list(executor.map(func, array))
        return res

    def smallerZero(self, list: list[int], c_list: list[int], i: int):
        res = self.decryptList(list, c_list, i)
        nHalf = self.publicKey.n//2
        for i, num in enumerate(res):
            if num > nHalf: # negative 
                res[i] = True
            else: # positive or zero
                res[i] = False
        return res

    def transitionIsEnabled(self, transitionColumn: int, marking: list[int]):
        places = []
        for i in range(len(self.consumptionMatrix.places)):
            if self.consumptionMatrix.c_matrix[i][transitionColumn] == -1:
                places.append(marking[i].value)
        res = self.publicKey.encrypt(1)
        if len(places) == 0:
            res = self.publicKey.encrypt(1)
        else:
            for place in places:
                res = EncryptedNumber(self.multNumbers(res.value, place), self.publicKey)
        return res.value
                
    
    def fire(self, candId: CandId, marking: list[int]):
        for i, place in enumerate(marking):
            marking[i] = EncryptedNumber(place, self.publicKey)
        activity = self.deanymMap[candId.id]
        activityColumn = None
        for key, val in self.incidenceMatrix.transitions.items():
            if key.name == activity:
                activityColumn = val
                break
        isEnabled = self.transitionIsEnabled(activityColumn, marking)
        firingVector = [place[activityColumn] for place in self.incidenceMatrix.a_matrix]
        for i, place in enumerate(firingVector):
            marking[i] = (marking[i] + self.publicKey.encrypt(place)).value
        return marking, isEnabled
    
    def multiplyEncryptedList(self, list: list[int]) -> list[int]:
        c_list = [None] * len(list)
        for i, num in enumerate(list):
            c = self.privateKeyShare.decrypt(EncryptedNumber(num, self.publicKey))
            c_list[i] = c
        otherOrg = None
        for org in self.orgPorts:
            if org.orgId != self.identifier:
                otherOrg = org.orgId
                break
        res = self.socket.sendToOrgWaitForResponse(otherOrg, Message(action="decryptList", data={
            "list": list,
            "c_list": c_list,
            "i": self.privateKeyShare.i
        }))
        res = res.data
        product = 1
        for num in res:
            product *= num
        res = self.publicKey.encrypt(product)
        return res.value
    
    def multNumbers(self, num1: int, num2: int):
        # decrypt num1 and num2
        c_list = [self.privateKeyShare.decrypt(EncryptedNumber(num1, self.publicKey)), self.privateKeyShare.decrypt(EncryptedNumber(num2, self.publicKey))]
        otherOrg = None
        for org in self.orgPorts:
            if org.orgId != self.identifier:
                otherOrg = org.orgId
                break
        res = self.socket.sendToOrgWaitForResponse(otherOrg, Message(action="decryptList", data={
            "list": [num1, num2],
            "c_list": c_list,
            "i": self.privateKeyShare.i
        }))
        res = res.data
        return (self.publicKey.encrypt(res[0] * res[1])).value
    
    def isEscapingEdge(self, candId: CandId, allowedTransitions: list[int]) -> list[int]:
        # deanym candId and get column of activity
        activity = self.deanymMap[candId.id]
        activityColumn = None
        for key, val in self.incidenceMatrix.transitions.items():
            if key.name == activity:
                activityColumn = val
                break
        # multiply allowedTransitions[activityColumn] with 0
        c = EncryptedNumber(allowedTransitions[activityColumn], self.publicKey)
        c = c * 0
        c += self.publicKey.encrypt(0) # obfuscate
        allowedTransitions[activityColumn] = c.value
        return allowedTransitions

    def isZero(self, value: int) -> bool:
        c = EncryptedNumber(value, self.publicKey)
        c_list = [self.privateKeyShare.decrypt(c)]
        otherOrg = None
        for org in self.orgPorts:
            if org.orgId != self.identifier:
                otherOrg = org.orgId
                break
        res = self.socket.sendToOrgWaitForResponse(otherOrg, Message(action="decryptList", data={
            "list": [value],
            "c_list": c_list,
            "i": self.privateKeyShare.i
        }))
        res = res.data
        return res[0] == 0
    
    def calculatePrecision(self, totalEscapingEdges: int, totalAllowedTransitions: int) -> float:
        totalEscapingEdges = EncryptedNumber(totalEscapingEdges, self.publicKey)
        totalAllowedTransitions = EncryptedNumber(totalAllowedTransitions, self.publicKey)
        c_list = [self.privateKeyShare.decrypt(totalEscapingEdges), self.privateKeyShare.decrypt(totalAllowedTransitions)]
        otherOrg = None
        for org in self.orgPorts:
            if org.orgId != self.identifier:
                otherOrg = org.orgId
                break
        res = self.socket.sendToOrgWaitForResponse(otherOrg, Message(action="decryptList", data={
            "list": [totalEscapingEdges.value, totalAllowedTransitions.value],
            "c_list": c_list,
            "i": self.privateKeyShare.i
        }))
        res = res.data
        return 1 - res[0] / res[1]

    def encryptCandidateTimestamps(self, timestamps: list[CandidateTimestamp]) -> list[CandidateTimestamp]:
        # Create a partial function with the additional arguments bound
        partial_func = partial(
            encryptCandidateTimestampsAPPLY, 
            publicKey=self.publicKey, 
            orgToInt=self.orgToInt
        )
        # Use the partial function in applyFuncToArray
        return self.applyFuncToArray(partial_func, timestamps)


    def partialDecryptCandidateTimestamps(
        self,
        timestamps: list[CandidateTimestamp],
        decryptTime: bool = True,
        decryptCand: bool = True
    ) -> list[CandidateTimestamp]:
        # Create a partial function with bound arguments
        partial_func = partial(
            partialDecryptCandidateTimestampsAPPLY,
            privateKeyShare=self.privateKeyShare,
            decryptTime=decryptTime,
            decryptCand=decryptCand
        )
        # Use the partial function in applyFuncToArray
        return self.applyFuncToArray(partial_func, timestamps)

        
    def decryptCandidateTimestamps(
        self,
        timestamps: list[CandidateTimestamp],
        decryptTime: bool = True,
        decryptCand: bool = True
    ) -> list[CandidateTimestamp]:
        # Create a partial function with bound arguments
        partial_func = partial(
            decryptCandidateTimestampsAPPLY,
            privateKeyShare=self.privateKeyShare,
            decryptTime=decryptTime,
            decryptCand=decryptCand,
            intToOrg=self.intToOrg
        )
        # Use the partial function in applyFuncToArray
        return self.applyFuncToArray(partial_func, timestamps)
    
    def reduceListToCandIdList(self, list: list[object]):
        res = []
        for obj in list:
            if obj is None:
                res.append(None)
            else:
                res.append(obj.candId)
        return res

    def showPrefixTree(self):
        pm4py.view_prefix_tree(self.prefixTree)

    def discoverPrefixTree(self):
        self.prefixTree = pm4py.discover_prefix_tree(self.log)
        # update prefix tree, set node.act = node.label
        stack = []
        stack.append(self.prefixTree)
        self.prefixTree.act = None
        self.prefixTree.label = None
        self.prefixTree.weight = self.numTraces
        while stack:
            node = stack.pop()
            for child in node.children:
                stack.append(child)
                child.act = child.label
        self.insertTimestampsToPrefixTree()
        return self.prefixTree

    def insertTimestampsToPrefixTree(self):
        stack = []
        stack.append(self.prefixTree)
        self.prefixTree.log = self.log
        while stack:
            node = stack.pop()
            for child in node.children:
                stack.append(child)
                cases = node.log.groupby(['case:concept:name']).nth(child.depth - 1).loc[node.log['concept:name'] == child.label]['case:concept:name'].values
                child.log = node.log[node.log['case:concept:name'].isin(cases)]
                child.cases = cases
                child.timestamps = self.logToTimestampList(node.log.groupby(['case:concept:name']).nth(child.depth - 1).loc[node.log['concept:name'] == child.label])
                child.weight = len(child.cases)

    def logToTimestampList(self, log: EventLog):
        timestamps = [None] * self.numTraces
        for row in log.iterrows():
            case = row[1]['case:concept:name']
            act = row[1]['concept:name']
            timestamp = row[1]['time:timestamp']
            timestamps[int(case)] = ActivityTimestamp(act, timestamp.value)
        return timestamps
    
    def obfuscateEncCandTimestamps(self, timestamps: list[CandidateTimestamp]) -> list[CandidateTimestamp]:
        for time in timestamps:
            if time is None:
                continue
            encTime = EncryptedNumber(time.time, self.publicKey)
            encTime.obfuscate()
            time.encTime = encTime.value
        return timestamps
    
    def addEncryptedListToCandTime(self, timestamps: list[CandidateTimestamp], encrypted_list: list[EncryptedNumber]) -> list[CandidateTimestamp]:
        # Create a partial function to pass additional arguments
        partial_func = partial(
            addEncryptedListToCandTimeAPPLY, 
            encrypted_list=encrypted_list, 
            publicKey=self.publicKey
        )
        # Prepare the data as an enumerated list of tuples (index, timestamp)
        indexed_timestamps = list(enumerate(timestamps))
        # Use the partial function in applyFuncToArray
        return self.applyFuncToArray(partial_func, indexed_timestamps)
    
    def multiplyMinusOne(self, list: list[EncryptedNumber]):
        for i, num in enumerate(list):
            if num is None:
                continue
            list[i] = -1 * num
        return list
    
    def generateRandomEncryptedList(self, rangeValues: tuple[int, int]) -> list[EncryptedNumber]:
        return [self.publicKey.encrypt(randint(rangeValues[0], rangeValues[1])) for _ in range(self.numTraces)]

    def generateRandomPermutation(self, n = None):
        if n is None:
            n = self.numTraces
        perm = list(range(n))  # Create a list [0, 1, 2, ..., n-1]
        random.shuffle(perm)   # Shuffle the list to create a random permutation
        return perm
    
    def inversePermutation(self, perm):
        n = len(perm)
        inverse = [0] * n  # Create an empty list to store the inverse
        for i, p in enumerate(perm):
            inverse[p] = i  # Map each value to its index in the original permutation
        return inverse
    
    def applyPermutation(self, lst, perm):
        if len(lst) != len(perm):
            raise ValueError("The length of the list and the permutation must be the same")
        
        # Apply the permutation by mapping each element to its new position
        permuted_list = [lst[i] for i in perm]
        
        return permuted_list
    


def encryptCandidateTimestampsAPPLY(timestamp: CandidateTimestamp, publicKey: PublicKey, orgToInt: dict[OrgId, int]):
        c = None
        if timestamp is None:
            encOrgId = publicKey.encrypt(0)
            encCandId = publicKey.encrypt(0)
            # value that is greater than all timestamps (but not greater than n once the random value is added)
            encTime = publicKey.encrypt(10000000000000000000000)
            c = CandidateTimestamp(
                EncCandId(encOrgId.value, encCandId.value),
                encTime.value
            )
        else:
            encOrgId = publicKey.encrypt(orgToInt[timestamp.candId.orgId])
            encCandId = publicKey.encrypt(timestamp.candId.id)
            encTime = publicKey.encrypt(timestamp.time)
            c = CandidateTimestamp(
                EncCandId(encOrgId.value, encCandId.value),
                encTime.value
            )
        return c

def addEncryptedListToCandTimeAPPLY(index_time: tuple[int, CandidateTimestamp], encrypted_list: list[EncryptedNumber], publicKey: PublicKey):
    i, time = index_time
    if time is None:
        return time  # Return the unchanged timestamp if it's None
    
    encTime = EncryptedNumber(time.time, publicKey)
    encTime = encTime + encrypted_list[i]
    time.encTime = encTime.value
    
    return time


def partialDecryptCandidateTimestampsAPPLY(
    timestamp: CandidateTimestamp,
    privateKeyShare,
    decryptTime: bool,
    decryptCand: bool
) -> CandidateTimestamp:
    if timestamp is None:
        return None

    resTime = timestamp.time
    resCandId = timestamp.candId

    if decryptTime:
        resTime = PartialEncTime(
            timestamp.time,
            privateKeyShare.decrypt(timestamp.time),
            privateKeyShare.i
        )

    if decryptCand:
        resCandId = PartialEncCandId(
            timestamp.candId.encOrgId,
            timestamp.candId.encId,
            privateKeyShare.decrypt(timestamp.candId.encId),
            privateKeyShare.decrypt(timestamp.candId.encOrgId),
            privateKeyShare.i
        )

    return CandidateTimestamp(resCandId, resTime)

def decryptCandidateTimestampsAPPLY(
    timestamp: CandidateTimestamp,
    privateKeyShare,
    decryptTime: bool,
    decryptCand: bool,
    intToOrg: dict[int, OrgId]
) -> CandidateTimestamp:
    resCandId = timestamp.candId
    resTime = timestamp.time
    decTime = timestamp.time
    decCandId = timestamp.candId.encId
    decOrgId = timestamp.candId.encOrgId

    if decryptTime:
        decTime = privateKeyShare.final_decrypt(
            [timestamp.time.partialEncTime, privateKeyShare.decrypt(timestamp.time.encTime)],
            [timestamp.time.partialEncI, privateKeyShare.i]
        )
        resTime = decTime

    if decryptCand:
        decCandId = privateKeyShare.final_decrypt(
            [timestamp.candId.partialEncId, privateKeyShare.decrypt(timestamp.candId.encId)],
            [timestamp.candId.partialEncI, privateKeyShare.i]
        )
        decOrgId = privateKeyShare.final_decrypt(
            [timestamp.candId.partialEncOrgId, privateKeyShare.decrypt(timestamp.candId.encOrgId)],
            [timestamp.candId.partialEncI, privateKeyShare.i]
        )
        decOrgId = intToOrg[decOrgId]
        resCandId = CandId(decOrgId, decCandId)

    return CandidateTimestamp(resCandId, resTime)