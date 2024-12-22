## main function
from coordinator import Coordinator
import pm4py
from customTypes.types import AnomTrie, OrgTrie, Act
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.conformance.alignments.petri_net import algorithm as computeAlignments
from organization import Organization
from pandas import DataFrame
from pm4py.objects.log.obj import Trace
from pm4py.objects.trie.obj import Trie
from pm4py.objects.petri_net.semantics import enabled_transitions, PetriNetSemantics
from pm4py.objects.petri_net.utils import incidence_matrix
from pm4py.objects.petri_net.utils.align_utils import get_visible_transitions_eventually_enabled_by_marking
from pm4py.algo.simulation.playout.petri_net import algorithm as petriNetSimulator
from pm4py.statistics.variants.log import get as get_variants_log
from utils import printProgressBar, store_anom_trie, load_anom_trie, mergeXesLogs
from time import time
from time import sleep
import os
import sys
import threading

def startTimer():
    return time()

def stopTimer(start: float):
    return time() - start

def viewModel():
    # prompt path of model
    path = input('Enter path to model: ')
    # read model
    model, initial_marking, final_marking = pm4py.read_pnml(path)
    # view model
    pm4py.view_petri_net(model, initial_marking, final_marking)

def exportModel():
    # prompt path of log
    path = input('Enter path to log: ')
    # prompt path to save model
    savePath = input('Enter path to save model: ')
    # read log
    log = pm4py.read_xes(path)
    # discover model
    model, initial_marking, final_marking = pm4py.discover_petri_net_ilp(log)
    # save model
    pm4py.write_pnml(model, initial_marking, final_marking, savePath)

def traceListToPrefixList(traces: list[Trace]) -> list[list[Act]]:
    res = []
    for trace in traces:
        prefix = []
        for act in list(trace):
            prefix.append(act['concept:name'])
        res.append(prefix)
    return res

def prefixListToTraceList(prefixes: list[list[Act]]) -> list[Trace]:
    res = []
    for prefix in prefixes:
        trace = []
        for act in prefix:
            trace.append({"concept:name": act})
        res.append(Trace(trace))
    return res

def trieToTraceList(root: OrgTrie | AnomTrie) -> tuple[list[Trace], list[int]]:
    stack = [root]
    prefixes = []
    prefixWeights = []

    def nodeToPrefix(node):
        prefix = []
        current = node
        while current is not None:
            if(current.act is not None):
                prefix = [{"concept:name": current.act}] + prefix
            current = current.parent
        return prefix

    while stack:
        current = stack.pop()
        if sum([child.weight for child in current.children]) < current.weight:
            prefixes.append(Trace(nodeToPrefix(current)))
            prefixWeights.append(current.weight - sum([child.weight for child in current.children]))
        stack += current.children
    
    return prefixes, prefixWeights
    

def tracesToFireSeq(traces: list[Trace], model: PetriNet, initial_marking: Marking, final_marking: Marking, weights = None) -> list[(str, str)]:

    # align prefixes with model
    alignments = []
    count = 0
    numUnfitTraces = 0
    printProgressBar(0, len(traces), prefix = 'Computing Firing Sequences:', suffix = 'Complete')
    for j, prefix in enumerate(traces):
        alignment = computeAlignments.apply_trace(prefix, model, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
        firingSeq = [(step[0][1], step[1][1]) for step in alignment['alignment']]
        alignments.append(firingSeq)
        count += 1
        # check if prefix is fit
        i = 0
        for (transitionName, act) in firingSeq:
            if act is None:
                continue
            if i == len(prefix) or (act != prefix[i]['concept:name']):
                numUnfitTraces += 1 * (weights[j] if weights is not None else 1)
                break
            i += 1
        printProgressBar(count, len(traces), prefix = 'Computing Firing Sequences:', suffix = 'Complete')
    print(numUnfitTraces, "of", len(traces), "traces are unfit")
    return alignments

def logToOrgTrie(path: str, numTraces: int = 0):
    org = Organization(identifier="org", orgPort=7778, coordinatorPort=7777, orgPorts=[], path=path, numTraces=numTraces, publicKey=None, privateKeyShare=None)
    return org.prefixTree

def fireSeqsToPrefixTree(lists: list[list[tuple[str, str | None]]]):
    prefixes = []
    for list in lists:
        prefix = []
        for tuple in list:
            if tuple[1] is not None:
                prefix.append(tuple[1])
        prefixes.append(prefix)
    root = Trie()
    for prefix in prefixes:
        # run through the trace, add each check if activity is already in children, if not add it
        current = root
        for act in prefix:
            found = False
            for child in current.children:
                if child.label == act:
                    current = child
                    found = True
                    break
            if not found:
                new = Trie()
                new.label = act
                new.parent = current
                current.children.append(new)
                current = new
    return root

def measureEscapingEdges(model: PetriNet, initial_marking: Marking, final_marking: Marking, fireSeqs, weights) -> float:

    def getTransition(name: str):
        for t in model.transitions:
            if t.name == name:
                return t        
        return None

    def getNextNode(node: Trie, act: str):
        for child in node.children:
            if child.label == act:
                return child
        return None
    
    rootPrefixTree = fireSeqsToPrefixTree(fireSeqs)
    
    numAllowedTasks = 0
    numOfEscapingEdges = 0
    count = 0
    printProgressBar(0, len(fireSeqs), prefix = 'Computing ETC:', suffix = 'Complete')
    for seq, weight in zip(fireSeqs, weights):
        currentNode = rootPrefixTree
        currentMarking = initial_marking
        for i, (transitionName, act) in enumerate([(None, None)] + seq):
            if i == 0:
                pass
            # silent transition
            elif act is None:
                # only update marking
                transition = getTransition(transitionName)
                fire = PetriNetSemantics().fire
                currentMarking = fire(model, transition, currentMarking)
                continue
            else:
                # get next marking
                transition = getTransition(transitionName)
                fire = PetriNetSemantics().fire
                try:
                    currentMarking = fire(model, transition, currentMarking)
                except:
                    print("Transition not found")
                    break
                # get next node
                currentNode = getNextNode(currentNode, act)

            allowedTasks = get_visible_transitions_eventually_enabled_by_marking(model, currentMarking)
            reflectedTasks = currentNode.children
            numAllowedTasks += len(allowedTasks) * weight
            numOfEscapingEdges += (len(allowedTasks) * weight - len(reflectedTasks) * weight)
        count += 1
        printProgressBar(count, len(fireSeqs), prefix = 'Replaying Traces:', suffix = 'Complete')

            
    return 1 - (float(numOfEscapingEdges) / float(numAllowedTasks))

def simulateModel(model: PetriNet, initial_marking: Marking, final_marking: Marking, numSimulatedTraces: int):
    simulated_log = petriNetSimulator.apply(model, initial_marking, final_marking, variant=petriNetSimulator.Variants.BASIC_PLAYOUT, parameters={petriNetSimulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: numSimulatedTraces})
    variants = get_variants_log.get_variants(simulated_log)
    return list(variants)

def classicalEtcPrecision(log: EventLog, model: PetriNet, initial_marking: Marking, final_marking: Marking):
    return pm4py.precision_token_based_replay(log, model, initial_marking, final_marking)

def classicalAlignEtcPrecision(trie: OrgTrie | AnomTrie, model: PetriNet, initial_marking: Marking, final_marking: Marking):
    traces, weights = trieToTraceList(trie)
    fireSeqs = tracesToFireSeq(traces, model, initial_marking, final_marking)
    return measureEscapingEdges(model, initial_marking, final_marking, fireSeqs, weights)

def getPrefix(tree: Trie) -> Trie:
    current = tree
    prefix = []
    while current.parent is not None:
        if current.label is not None:
            prefix = [current.label] + prefix
        current = current.parent
    return prefix

def prefixToFireSeq(prefix: list[Act], model: PetriNet, preTransition, postTransition) -> list[(str, str)]:
    res = [preTransition]
    for act in prefix:
        for t in model.transitions:
            if t.name == act:
                res.append(t.name)
                break
    res.append(postTransition)
    return res

def isValidFireSeq(fireSeq: list[(str, str)], model: PetriNet, initial_marking: Marking, final_marking: Marking) -> bool:

    def getTransition(transitionName: str):
        for t in model.transitions:
            if t.name == transitionName:
                return t
        return None
    currentMarking = initial_marking
    currentMarking = PetriNetSemantics().fire(model, getTransition(fireSeq[0]), currentMarking)
    for i, transitionName in enumerate(fireSeq[1:len(fireSeq)-1]):
        enabledTransitions = get_visible_transitions_eventually_enabled_by_marking(model, currentMarking)
        
        isEnabled = False
        for enabledTransition in enabledTransitions:
            if enabledTransition.name == transitionName:
                isEnabled = True
                break
        if not isEnabled:
            return False
        
        else:
            currentMarking = PetriNetSemantics().fire(model, getTransition(transitionName), currentMarking)
    currentMarking = PetriNetSemantics().fire(model, getTransition(fireSeq[-1]), currentMarking)
    print(currentMarking)
    return True
        

def main(): 
    # model, initial_marking, final_marking = pm4py.read_pnml('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/road/road-prepared-ilp.pnml')
    # model, initial_marking, final_marking = pm4py.read_pnml('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/sepsis/sepsis_model_ilp.pnml')
    # coordinator = Coordinator(7777, [
    #     ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2plitRoad(random1)/2plitRoad(random1)_org1.xes"),
    #     ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2plitRoad(random1)/2plitRoad(random1)_org1.xes")
    # ], 150370)
    # log = pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/road/road-prepared.xes')
    # test = pm4py.discover_prefix_tree(log)
    # print(countNodes(test))
    # coordinator = Coordinator(7777, [
    #     ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitRoad20(random1)/2splitRoad20(random1)_org1.xes"),
    #     ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitRoad20(random1)/2splitRoad20(random1)_org2.xes")
    # ], 30074)
    tree, deanymLists = load_anom_trie('modelsAndLogs/prefixTrees/sepsis/2splitSepsis')
    print(treeBranchingFactor(tree))
    # coordinator.mergedPrefixTree = tree
    # for org in coordinator.orgs:
    #     org.deanymMap = deanymLists[org.identifier]
    # timer = startTimer()
    # print(coordinator.replayLog(model, initial_marking, final_marking, [], "■", sampleSize=0.25))
    # print(coordinator.mergeTrees())
    # print(stopTimer(timer))
    # coordinator.close()
    # coordinator.close()


    # coordinator.deanymAnomTrie(coordinator.mergedPrefixTree, fullDeanym=True)
    # traces, weights = trieToTraceList(coordinator.mergedPrefixTree)
    # prefixes = traceListToPrefixList(traces)
    # fireSeqs = [prefixToFireSeq(prefix, model, "▶", "■") for prefix in prefixes]
    # countValid = 0
    # countInvalid = 0
    # for i, fireSeq in enumerate(fireSeqs):
    #     if not isValidFireSeq(fireSeq, model, initial_marking, final_marking):
    #         countInvalid += 1 * weights[i]
    #     else:
    #         countValid += 1 * weights[i]
    # print("Valid: ", countValid)
    # print("Invalid: ", countInvalid)
    # coordinator.close()


def treeBranchingFactor(tree: Trie):
    # returns the maximum and average branching factor of a tree
    stack = [tree]
    maxBranchingFactor = 0
    totalBranchingFactor = 0
    numNodes = 0
    while stack:
        current = stack.pop()
        numNodes += 1
        totalBranchingFactor += len(current.children)
        if len(current.children) > maxBranchingFactor:
            maxBranchingFactor = len(current.children)
        stack += current.children
    return maxBranchingFactor, totalBranchingFactor/numNodes

def are_trees_same(tree1, tree2, path="root"):
    # Base case: if both nodes are None, they are identical at this level
    if not tree1 and not tree2:
        return 0, 0
    # If only one of the nodes is None, consider this as a non-equal node
    if not tree1 or not tree2:
        print(f"Mismatch at path: {path}")
        return 1, 1
    
    # Initialize counters
    checked_nodes = 1
    non_equal_nodes = 0
    
    # Check the current node values
    if tree1.label != tree2.act:
        non_equal_nodes = 1
        print(f"Mismatch at path: {path} (tree1.label='{tree1.label}', tree2.act='{tree2.act}')")
    
    # Check if both nodes have the same number of children
    if len(tree1.children) != len(tree2.children):
        non_equal_nodes += 1
        print(f"Mismatch in number of children at path: {path} (tree1.children={len(tree1.children)}, tree2.children={len(tree2.children)})")
        prefix = getPrefix(tree1)
        prefix = prefix
        return checked_nodes, non_equal_nodes
    
    # Sort children by label and act to align corresponding children
    sorted_children1 = sorted(tree1.children, key=lambda x: x.label)
    sorted_children2 = sorted(tree2.children, key=lambda x: x.act)
    
    # Recursively check each child and accumulate counts
    for i, (c1, c2) in enumerate(zip(sorted_children1, sorted_children2)):
        child_path = f"{path} -> child[{i}]"
        child_checked, child_non_equal = are_trees_same(c1, c2, child_path)
        checked_nodes += child_checked
        non_equal_nodes += child_non_equal
    
    return checked_nodes, non_equal_nodes

def countNodes(trie: Trie):
    stack = [trie]
    count = 0
    while stack:
        current = stack.pop()
        count += 1
        stack += current.children
    return count

def mergeTreesInFolder(pathToFolder: str, numCases: int, firstPort: int = 7777, naiveCompare: bool = False, oneIteration: bool = False):
    # find files in folder
    files = os.listdir(pathToFolder)
    # merge trees
    config = []
    for i, file in enumerate(files):
        config.append((f"org{i+1}", firstPort+i+1, pathToFolder + "/" + file))
    coordinator = Coordinator(firstPort, config, numCases)
    timer = startTimer()
    coordinator.mergeTrees(naive=naiveCompare, oneIteration=oneIteration)
    print("-----------------------------")
    print("Path to folder: ", pathToFolder)
    print("Seconds to merge trees: ", stopTimer(timer))
    print("Naive compare: ", naiveCompare)
    print("One iteration: ", oneIteration)
    print("Number of nodes in merged tree: ", countNodes(coordinator.mergedPrefixTree))
    print("-----------------------------")
    # append output to file evaluation/results.txt
    with open("evaluation/results.txt", "a") as file:
        file.write("-----------------------------\n")
        file.write("Path to folder: " + pathToFolder + "\n")
        file.write("Seconds to merge trees: " + str(stopTimer(timer)) + "\n")
        file.write("Naive compare: " + str(naiveCompare) + "\n")
        file.write("One iteration: " + str(oneIteration) + "\n")
        file.write("Number of nodes in merged tree: " + str(countNodes(coordinator.mergedPrefixTree)) + "\n")
        file.write("-----------------------------\n")
    coordinator.close()
    return coordinator.mergedPrefixTree

def etcPeval(pathToPrefixTree: str, pathToModel: str, sampleSize: float, preTransition: str, postTransition: str, oneIteration: bool = False):
    model, initial_marking, final_marking = pm4py.read_pnml(pathToModel)
    tree, deanymLists = load_anom_trie(pathToPrefixTree)
    coordinator = Coordinator(7767, [
        ("org1", 7768, None),
        ("org2", 7769, None)
    ], 1)
    coordinator.mergedPrefixTree = tree
    for org in coordinator.orgs:
        org.deanymMap = deanymLists[org.identifier]
    timer = startTimer()
    if preTransition == "None":
        preTransition = []
    if postTransition == "None":
        postTransition = []
    print(coordinator.replayLog(model, initial_marking, final_marking, preTransition, postTransition, sampleSize=sampleSize, oneIteration=oneIteration))
    print("-----------------------------")
    print("Path to folder: ", pathToPrefixTree)
    print("Seconds to eval etcP: ", stopTimer(timer))
    print("Sample Size: ", sampleSize)
    print("One iteration: ", oneIteration)
    print("-----------------------------")
    # append output to file evaluation/results.txt
    with open("evaluation/results.txt", "a") as file:
        file.write("-----------------------------\n")
        file.write("Path to folder: " + pathToPrefixTree + "\n")
        file.write("Seconds to eval etcP: " + str(stopTimer(timer)) + "\n")
        file.write("Sample Size: " + str(sampleSize) + "\n")
        file.write("One iteration: " + str(oneIteration) + "\n")
        file.write("-----------------------------\n")
    coordinator.close()

if __name__ == "__main__":
    
    if len(sys.argv) > 1 and sys.argv[1] == "mergeTreesInFolder":
        print(sys.argv)
        mergeTreesInFolder(sys.argv[2], int(sys.argv[3]), naiveCompare=("naive" in sys.argv), oneIteration=("oneIteration" in sys.argv))

    elif len(sys.argv) > 1 and sys.argv[1] == "etcP":
        print(sys.argv)
        etcPeval(sys.argv[2], sys.argv[3], float(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]), oneIteration=("oneIteration" in sys.argv))

    else:
        main()


# horizontal test split
# Coordinator(7777, [
#         ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/testHorizontalSplit/org1.xes"),
#         ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/testHorizontalSplit/org2.xes")
#     ], 1050)

# sepsis (2 split) (0.25 removed)
# Coordinator(7777, [
#         ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitSepsisQuarter/2splitSepsisQuarter_org1.xes"),
#         ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitSepsisQuarter/2splitSepsisQuarter_org2.xes")
#     ], 262)

# # sepsis (2 split) (0.5 removed)
# Coordinator(7777, [
#         ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitSepsisHalf/2splitSepsisHalf_org1.xes"),
#         ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitSepsisHalf/2splitSepsisHalf_org2.xes")
#     ], 525)

# # sepsis (2 split) (0.75 removed)
# Coordinator(7777, [
#         ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitSepsis75/2splitSepsis75_org1.xes"),
#         ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitSepsis75/2splitSepsis75_org2.xes")
#     ], 787)

# # sepsis (2 split) (6641 nodes)
# Coordinator(7777, [
#         ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitSepsis/2splitSepsis_org1.xes"),
#         ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/2splitSepsis/2splitSepsis_org2.xes")
#     ], 1050)

# # sepsis (3 split) (6641 nodes) 
# Coordinator(7777, [
#         ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/3splitSepsis/3splitSepsis_org1.xes"),
#         ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/3splitSepsis/3splitSepsis_org2.xes"),
#         ("org3", 7780, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/3splitSepsis/3splitSepsis_org3.xes")
#     ], 1050)

# # sepsis (4 split) (6641 nodes)
# Coordinator(7777, [
#         ("org1", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/4splitSepsis/4splitSepsis_org1.xes"),
#         ("org2", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/4splitSepsis/4splitSepsis_org2.xes"),
#         ("org3", 7780, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/4splitSepsis/4splitSepsis_org3.xes"),
#         ("org4", 7781, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/evaluation/4splitSepsis/4splitSepsis_org4.xes")
#     ], 1050)







## old

# sepsis (2 split) (6641 nodes) {'Clinical Procedures': ['Release A', 'Release B', 'Release C', 'Release D', 'Release E', 'Leucocytes', 'CRP', 'LacticAcid', 'IV Liquid', 'IV Antibiotics'], 'Patient Management': ['Return ER', 'Admission NC', 'Admission IC', 'ER Registration', 'ER Triage', 'ER Sepsis Triage']}
# Coordinator(7777, [
#         ("Clinical Procedures", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/sepsis/activity_split/activity_split_clinicalProcedures.xes"),
#         ("Patient Management", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/sepsis/activity_split/activity_split_patientManagement.xes")
#     ], 1050)

# Hospital_billing (activity_split) (3933 nodes) {'finance': ['FIN', 'BILLED', 'DELETE', 'STORNO', 'REJECT', 'CODE NOK', 'SET STATUS', 'CODE ERROR'], 'patient': ['NEW', 'RELEASE', 'CODE OK', 'REOPEN', 'CHANGE DIAGN', 'CHANGE END', 'MANUAL', 'JOIN-PAT', 'ZDBC_BEHAN', 'EMPTY']}
# Coordinator(7777, [
#         ("finance", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/Hospital_billing/activity_split/activity_split_finance_administration.xes"),
#         ("patient", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/Hospital_billing/activity_split/activity_split_patient_diagnosis.xes")
#     ], 100000, activityPartition=activityPartition)

# running example
# Coordinator(7777, [
#         ("orgA", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/temp/orgA.xes"),
#         ("orgB", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/temp/orgB.xes")
#     ], 4)

# BPI2017-Final (split) (has 22006 nodes)
# Coordinator(7777, [
#         ("orgA", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/split/BPI2017_split_orgA.xes"),
#         ("orgB", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/split/BPI2017_split_orgB.xes")
#     ], 31508)

# BPI2017-Final (smaller_split) (has 1903 nodes)
# Coordinator(7777, [
#         ("orgA", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/smaller_split/BPI2017_split_orgA.xes"),
#         ("orgB", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/smaller_split/BPI2017_split_orgB.xes")
#     ], 31508)

# BPI2017-Final (fourfold_split)
# Coordinator(7777, [
#         ("orgA", 7778, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/fourfold_split/BPI2017_split_orgA.xes"),
#         ("orgB", 7779, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/fourfold_split/BPI2017_split_orgB.xes"),
#         ("orgC", 7780, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/fourfold_split/BPI2017_split_orgC.xes"),
#         ("orgD", 7781, "/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/fourfold_split/BPI2017_split_orgD.xes")
#     ], 31508)