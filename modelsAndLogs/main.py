from typing import Mapping
import pm4py
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.exporter.xes import exporter as logExporter
from pandas import concat
import random

# # read csv
# path = '/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/BPI2017-filtered.csv' # Enter path to the csv file
# data = pd.read_csv(path)
# cols = ['bla1', 'bla2','bla3', 'concept:name', 'bla4', 'bla5', 'bla6','time:timestamp','case:concept:name']
# data.columns = cols
# data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
# data['concept:name'] = data['concept:name'].astype(str)
# log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)





def splitLogPrompt():
    # prompt path to original log
    originalLogPath = input('Enter path to original log: ')
    log = pm4py.read_xes(originalLogPath)
    # random.seed(0)
    # prompt the user for the percentage of the log to keep
    keepOriginal = float(input('Enter percentage of the log to keep (0-1): '))
    # Remove keepOriginal percent of the log (randomly selected)
    caseIds = log['case:concept:name'].unique()
    # randomly select caseIds to keep (keepOriginal percent)
    keep = random.sample(list(caseIds), int(len(caseIds)*keepOriginal))
    log = log[log['case:concept:name'].isin(keep)]
    # reset index
    log.reset_index(drop=True, inplace=True)
    # also reset case:concept:name
    case_id = log['case:concept:name'].unique()
    case_dict = {case_id[i]: i for i in range(len(case_id))}
    log['case:concept:name'] = log['case:concept:name'].apply(lambda x: case_dict[x])
    # prompt the base name of the split logs
    baseName = input('Enter base name for split logs: ')
    # prompt the user for path where split logs should be saved
    path = input('Enter path to save split logs: ')
    # prompt the user for name of organization until the user enters 'done'
    orgs = []
    while True:
        org = input('Enter organization name (enter "done" to finish): ')
        if org == 'done':
            break
        orgs.append(org)
    # print list of unique activities in the log
    print("Unique activities in the log ("+str(len(log['concept:name'].unique()))+"): ")
    uniqueActivities = log['concept:name'].unique()
    strActivities = ""
    for i in range(len(uniqueActivities)):
        strActivities += uniqueActivities[i] + ", "
    print(strActivities)
    print()
    # prompt the user for activities of each organization
    # activities seperated by comma
    partition = []
    # ask if partition should be randomly calculated
    randomPartition = input('Randomly partition activities? (y/n): ')   
    if randomPartition == 'y':
        random.shuffle(uniqueActivities)
        partition = [uniqueActivities[i::len(orgs)] for i in range(len(orgs))]
    else:
        for org in orgs:
            actions = input(f'Enter activities for organization {org} (seperate by comma): ')
            actions = actions.split(',')
            actions = [action.strip() for action in actions]
            partition.append(actions)
    # split the log and save the split logs
    split = splitEventLog(log, partition, [f'{path}/{baseName}_{org}.xes' for org in orgs])


def splitEventLog(log: EventLog, partition: list[list[str]], paths: list[str] = None) -> list[EventLog]:
    res = []
    for i, actions in enumerate(partition):
        part = log.loc[log['concept:name'].isin(actions)]
        part.reset_index(drop=True, inplace=True)
        if paths is not None:
            logExporter.apply(part, paths[i])
        res.append(part)
    return res

def prepareXesLog(path: str, overwrite: bool = False, savePath: str = None):
    if savePath is not None and not overwrite:
        raise Exception('savePath and overwrite cannot be True at the same time')
    log = pm4py.read_xes(path)
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    log['concept:name'] = log['concept:name'].astype(str)
    # the column 'case:concept:name' is the case identifier
    case_id = log['case:concept:name'].unique()
    case_dict = {case_id[i]: i for i in range(len(case_id))}
    log['case:concept:name'] = log['case:concept:name'].apply(lambda x: case_dict[x])
    data = log[['concept:name', 'time:timestamp', 'case:concept:name']]
    log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    if savePath is not None:
        logExporter.apply(log, savePath)
    elif overwrite:
        logExporter.apply(log, path)

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


if __name__ == '__main__':

    splitLogPrompt()
