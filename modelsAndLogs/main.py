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





def splitLogPrompt(log: EventLog):
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

    # splitLogPrompt(pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2018-department/BPI2018-department-prepared.xes'))
    # splitLogPrompt(pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/sepsis/sepsis_prepared_time_fixed.xes'))
    splitLogPrompt(pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/road/road-prepared.xes'))
    # splitLogPrompt(pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/2012-prepared.xes'))


    # # read csv
    # data = pd.read_csv('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/BPI2017-Final-prepared.csv')
    # cols = ['concept:name', 'time:timestamp', 'case:concept:name']
    # data.columns = cols
    # data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
    # data['concept:name'] = data['concept:name'].astype(str)
    # log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    # trie = pm4py.discover_prefix_tree(log)

    # log1 = pd.read_csv('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/fourfold_split/BPI2017_split_orgA.xes')
    # log2 = pd.read_csv('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/fourfold_split/BPI2017_split_orgB.xes')
    # log3 = pd.read_csv('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/fourfold_split/BPI2017_split_orgC.xes')
    # log4 = pd.read_csv('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/fourfold_split/BPI2017_split_orgD.xes')
    
    # log = mergeXesLogs([log1, log2, log3, log4])
    # trie = pm4py.discover_prefix_tree(log)
    # # count nodes in trie
    # current = trie
    # stack = []
    # count = 0
    # while True:
    #     count += 1
    #     stack += current.children
    #     if len(stack) == 0:
    #         break
    #     current = stack.pop()
    # print(count)


    # # read csv
    # path = '/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/BPI2017-Final-prepared.csv' # Enter path to the csv file
    # data = pd.read_csv(path)
    # cols = ['bla1', 'bla2','bla3','bla4', 'concept:name', 'bla4', 'time:timestamp','bla5','bla6','bla7','bla8','bla9','bla10','bla11','bla12','bla13','bla14','bla15','bla16','bla17','bla18','bla19','case:concept:name']
    # data.columns = cols
    # data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
    # data['concept:name'] = data['concept:name'].astype(str)
    # # the column 'case:concept:name' is the case identifier
    # # it should be reorganized, such that the case identifiers are integers from 0 to the number of cases
    # case_id = data['case:concept:name'].unique()
    # case_dict = {case_id[i]: i for i in range(len(case_id))}
    # data['case:concept:name'] = data['case:concept:name'].apply(lambda x: case_dict[x])
    # # all columns beside 'concept:name', 'time:timestamp' and 'case:concept:name' sould be dropped
    # data = data[['concept:name', 'time:timestamp', 'case:concept:name']]
    # # export the data to a new csv file
    # # data.to_csv('BPI2017-Final-prepared.csv', index=False)
    # log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)


    # prepare sepsis
    # log = pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/sepsis/split(non privat)/Sepsis Cases - Event Log.xes')
    # # read csv
    # log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    # log['concept:name'] = log['concept:name'].astype(str)
    # # the column 'case:concept:name' is the case identifier
    # # it should be reorganized, such that the case identifiers are integers from 0 to the number of cases
    # case_id = log['case:concept:name'].unique()
    # case_dict = {case_id[i]: i for i in range(len(case_id))}
    # log['case:concept:name'] = log['case:concept:name'].apply(lambda x: case_dict[x])
    # # all columns beside 'concept:name', 'time:timestamp' and 'case:concept:name' sould be dropped
    # data = log[['concept:name', 'time:timestamp', 'case:concept:name']]
    # log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    # # fix time
    # import pandas as pd

    # log['time:timestamp'] = log.groupby('case:concept:name').apply(
    #     lambda group: group['time:timestamp'] + pd.to_timedelta(range(len(group)), unit='ms')
    # ).reset_index(level=0, drop=True)
    #
    # # export to xes
    # pm4py.write_xes(log, 'Sepsis-prepared.xes')

    # prepare hospital billing log
    # log = pm4py.read_xes('/Users/Studium/Downloads/Hospital Billing - Event Log_1_all/Hospital Billing - Event Log.xes')
    # log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    # log['concept:name'] = log['concept:name'].astype(str)
    # # the column 'case:concept:name' is the case identifier
    # # it should be reorganized, such that the case identifiers are integers from 0 to the number of cases
    # case_id = log['case:concept:name'].unique()
    # case_dict = {case_id[i]: i for i in range(len(case_id))}
    # log['case:concept:name'] = log['case:concept:name'].apply(lambda x: case_dict[x])
    # # all columns beside 'concept:name', 'time:timestamp' and 'case:concept:name' sould be dropped
    # data = log[['concept:name', 'time:timestamp', 'case:concept:name']]
    # log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    # # export to xes
    # pm4py.write_xes(log, 'Hospital_billing-prepared.xes')

    # prepare bpi2018-department
    # log = pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/rawLogs/BPI2018-department.xes')
    # log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    # log['concept:name'] = log['concept:name'].astype(str)
    # # the column 'case:concept:name' is the case identifier
    # # it should be reorganized, such that the case identifiers are integers from 0 to the number of cases
    # case_id = log['case:concept:name'].unique()
    # case_dict = {case_id[i]: i for i in range(len(case_id))}
    # log['case:concept:name'] = log['case:concept:name'].apply(lambda x: case_dict[x])
    # # all columns beside 'concept:name', 'time:timestamp' and 'case:concept:name' sould be dropped
    # data = log[['concept:name', 'time:timestamp', 'case:concept:name']]
    # log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    # # export to xes
    # pm4py.write_xes(log, 'BPI2018-department-prepared.xes')

    # prepare road
    # Read the XES file
    # log = pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/road/road-prepared.xes')
    # # Convert EventLog to a DataFrame for manipulation
    # df = pm4py.convert_to_dataframe(log)
    # # Ensure timestamps are in datetime format
    # df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    # # Ensure 'concept:name' is a string
    # df['concept:name'] = df['concept:name'].astype(str)
    # # Reorganize case identifiers
    # case_id = df['case:concept:name'].unique()
    # case_dict = {case_id[i]: i for i in range(len(case_id))}
    # df['case:concept:name'] = df['case:concept:name'].apply(lambda x: case_dict[x])
    # # Keep only the necessary columns
    # df = df[['concept:name', 'time:timestamp', 'case:concept:name']]
    # # Fix time: Adjust timestamps within each case by adding millisecond increments
    # df['time:timestamp'] = df.groupby('case:concept:name').apply(
    #     lambda group: group['time:timestamp'] + pd.to_timedelta(range(len(group)), unit='ms')
    # ).reset_index(level=0, drop=True)
    # # Convert back to EventLog
    # log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
    # # Export to XES
    # pm4py.write_xes(log, 'road-prepared.xes')

    # prepare BPIC 2012
    # log = pm4py.read_xes('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/rawLogs/BPI_Challenge_2012.xes')
    # log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    # log['concept:name'] = log['concept:name'].astype(str)
    # # the column 'case:concept:name' is the case identifier
    # # it should be reorganized, such that the case identifiers are integers from 0 to the number of cases
    # case_id = log['case:concept:name'].unique()
    # case_dict = {case_id[i]: i for i in range(len(case_id))}
    # log['case:concept:name'] = log['case:concept:name'].apply(lambda x: case_dict[x])
    # # all columns beside 'concept:name', 'time:timestamp' and 'case:concept:name' sould be dropped
    # data = log[['concept:name', 'time:timestamp', 'case:concept:name']]
    # log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    # # export to xes
    # pm4py.write_xes(log, '2012-prepared.xes')




# print(len(pm4py.get_variants(log[log["doctype"] == "Entitlement application"])))
# tempModel = pm4py.discover_petri_net_inductive(log[log["doctype"] == "Entitlement application"])
# pm4py.view_petri_net(tempModel[0])

# ['Payment application' 'Entitlement application' 'Parcel document'
#  'Control summary' 'Reference alignment' 'Department control parcels'
#  'Inspection' 'Geo parcel document']