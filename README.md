# federated-PM-etcP

This repository contains tools to construct to compute the `merged prefix tree` and `ETC Precision` in a federated setting.

## Event Log Splitting

Run the script `modelsAndLogs/main.py` to split an event log.

### Input Logs
- Original event logs are located at:
  - `modelsAndLogs/preparedLogs/road/road-prepared.xes`
  - `modelsAndLogs/preparedLogs/sepsis/sepsis_prepared.xes`

### Example Output
- Examples of split logs are provided in the following folders:
  - `evaluation/2splitRoad5`
  - `evaluation/2splitSepsis`

## Merging Prefix Trees

To compute the merged prefix tree, use the command:

```bash
mergedPrefixTree/main.py mergeTreesInFolder [path to folder with event logs] [number of cases in log] [optional: "naive" (for naive implementation)] [optional: "store" (merged prefix tree is stored)]
```

### Examples
- Using the standard implementation and storing the result:
  ```bash
  mergedPrefixTree/main.py mergeTreesInFolder "evaluation/2splitRoad5" 7518 store
  ```
- Using the naive implementation and storing the result:
  ```bash
  mergedPrefixTree/main.py mergeTreesInFolder "evaluation/2splitRoad5" 1050 store naive
  ```

## Calculating ETC Precision

To compute the ETC Precision metric, use the command:

```bash
mergedPrefixTree/main.py etcP [path to prefix tree] [path to model] [percentage of variants that should be replayed (float 0-1)] [optional transition to be fired before execution] [optional transition to be fired after execution]
```

### Example
```bash
mergedPrefixTree/main.py etcP "evaluation/2splitSepsis/2splitSepsis.json" "modelsAndLogs/preparedLogs/sepsis/sepsis_model_ilp.pnml" 0.1 "▶" "■"
```

