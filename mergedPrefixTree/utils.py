from typing import Mapping
from pm4py.objects.trie.obj import Trie
from customTypes.types import AnomTrie, OrgId, Act, CandId
from pm4py.objects.log.obj import EventLog
from pandas import concat
import json
import sys


def serealizeTriePrefix(node: Trie):
    res = []
    while node.label is not None:
        res = [node.label] + res
        node = node.parent
    return res

def serealizeAnomTriePrefix(node: AnomTrie):
    res = []
    while node.candId is not None:
        res = [node.candId] + res
        node = node.parent
    return res

def mergeXesLogs(logs: list[EventLog], maps: list[Mapping[str, str]]| None = None):
    if maps is not None:
        for log in logs:
            log['case:concept:name'] = log['case:concept:name'].map(maps[logs.index(log)])
    merged = concat(logs)
    merged.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
    # merged.reset_index(drop=True, inplace=True)

    # map values back to normal
    if maps is not None:
        for log in logs:
            inv_map = {v: k for k, v in maps[logs.index(log)].items()}
            log['case:concept:name'] = log['case:concept:name'].map(inv_map)

    return merged

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
       # Calculate percent completion
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    # Calculate the filled length of the bar
    filledLength = int(length * iteration // total)
    # Create the bar string with fill and remaining space
    bar = fill * filledLength + '-' * (length - filledLength)
    # Print the progress bar
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()  # Ensure immediate output to the terminal

    # Print a newline when complete
    if iteration == total:
        print()

def serialize_anom_trie(node: AnomTrie):
    """ Recursively serialize the TrieNode and its children into a dictionary. """
    return {
        'act': node.act,
        'candId': {
            'id': node.candId.id,
            'orgId': node.candId.orgId,
        } if node.candId is not None else None,
        'weight': node.weight,
        'weightFrac': node.weightFrac,
        'label': node.label,
        'sensitiveActivities': node.sensitiveActivities,
        'children': [serialize_anom_trie(child) for child in node.children],
    }

def store_anom_trie(root, path, deanymLists: dict[OrgId, dict[int, Act]] = None):
    """ Store the Trie in a JSON file. """
    with open(path, 'w') as f:
        json.dump({'trie': serialize_anom_trie(root), 'deanymLists': deanymLists}, f)

def deserialize_anom_trie(data, parent=None):
    """ Recursively deserialize the dictionary back into a TrieNode. """
    node = AnomTrie()
    node.act = data['act']
    node.parent = parent
    if data['candId'] is not None:
        node.candId = CandId(data['candId']['orgId'], data['candId']['id'])
        # node.candId = CandId(data['candId']['orgId'], data['candId']['id'], data['candId']['weight'])
    node.weight = data['weight']
    node.weightFrac = data['weightFrac']
    node.label = data['label']
    if 'sensitiveActivities' in data:
        node.sensitiveActivities = data['sensitiveActivities']
    node.children = []
    for child in data['children']:
        node.children.append(deserialize_anom_trie(child, node))
    return node

def load_anom_trie(path):
    """ Load the Trie from a JSON file. """
    with open(path, 'r') as f:
        data = json.load(f)
    # convert the keys in deanymLists to integers (currently: dict[str, dict[str, str]], should be dict[str, dict[int, str]])
    for org in data['deanymLists']:
        data['deanymLists'][org] = {int(k): v for k, v in data['deanymLists'][org].items()}
    return deserialize_anom_trie(data['trie']), data['deanymLists']