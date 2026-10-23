# Follow-up experiment: does broader training fix the imbalance weakness?

The first experiment found that prediction-informed control was outperformed by
reactive control under sustained directional imbalance, a demand structure that
did not appear in the training profiles. This follow-up broadens the training
data to include such structures, retrains, and re-evaluates.

**The six evaluation scenarios are not regenerated and not touched.** The new
training profiles use new profile ids (60-71, 80-85, 90-95) and new seeds
(6000-6011, 7000-7005, 8000-8005), generated independently. Training on demand
structures of a similar kind to those evaluated is generalisation coverage;
training on the evaluation runs themselves would be leakage, and is not done.

Run from the project root, with the environment active.

## 1. Build the broadened partitions

```bash
python3 demand/build_partitions.py --broad
```

24 training profiles (12 balanced as before, 12 spanning axis imbalance from
0.06 to 0.36), 12 validation, 14 test. About 40,800 movement records.

## 2. Extract the datasets

```bash
python3 scripts/extract_dataset.py \
    --partitions demand/partitions_broad \
    --outdir data_broad
```

## 3. Train the broadened model

```bash
python3 scripts/train_lstm.py \
    --datadir data_broad \
    --outdir models_broad \
    --repeats 3
```

Keep the reported accuracy figures. They are a result in their own right: a
model trained on more varied demand may be slightly less accurate on average
while being more useful in conditions it previously failed on.

## 4. Re-run the experiment with the broadened model

```bash
python3 scripts/run_experiment.py \
    --jobs 8 \
    --model-dir "$(pwd)/models_broad" \
    --outdir results/experiment_broad
```

## 5. Analyse

```bash
python3 scripts/analyse_experiment.py --outdir results/experiment_broad
```

## What to compare

The question is narrow: **in the imbalanced scenario, does the broadened model
close the gap to the reactive controller?**

Original result, imbalanced, B versus C:

| metric | reactive | predictive | gap |
|---|---|---|---|
| waiting time | 51.6 s | 77.1 s | 25.5 s |
| time loss | 69.0 s | 100.7 s | 31.7 s |
| duration | 107.0 s | 138.7 s | 31.7 s |

Also check the other five scenarios did not get worse. A model that fixes
imbalance while degrading elsewhere is a trade, not a fix, and would need
reporting as such.

## Reporting

Both experiments go in the dissertation, in order. The first identifies the
limitation; the second tests whether broader training addresses it. If the gap
does not close, that is a firmer finding: the limitation is not simply a matter
of training breadth.
