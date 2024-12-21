import pm4py
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.exporter.xes import exporter as logExporter 



def splitLogPrompt(log: EventLog):
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
    print("Unique activities in the log: ")
    uniqueActivities = log['concept:name'].unique()
    strActivities = ""
    for i in range(len(uniqueActivities)):
        strActivities += uniqueActivities[i] + ", "
    print(strActivities)
    print()
    # prompt the user for activities of each organization
    # activities seperated by comma
    partition = []
    for org in orgs:
        actions = input(f'Enter activities for organization {org} (seperate by comma): ')
        actions = actions.split(',')
        actions = [action.strip() for action in actions]
        partition.append(actions)
    print(partition)
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


if __name__ == '__main__':

    # read csv
    data = pd.read_csv('/Users/Studium/Desktop/bachelor_thesis_restart/bachelor_thesis_restart/modelsAndLogs/preparedLogs/BPI2017-Final/BPI2017-Final-prepared.csv')
    cols = ['concept:name', 'time:timestamp', 'case:concept:name']
    data.columns = cols
    data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
    data['concept:name'] = data['concept:name'].astype(str)
    splitLogPrompt(data)
