# LSTM-Based Traffic Flow Prediction and Adaptive Traffic Signal Control Using SUMO

Code, configuration, trained models and results for the MSc dissertation of
Opeyemi Simeon Adeyeye (2025/B/MIT/1229), Miva Open University.

The project compares three signal controllers at a single simulated four-arm junction
in SUMO: a Webster fixed-time plan (A), a reactive controller that allocates green from
the last cycle's detector counts (B), and a prediction-informed controller that allocates
green from an LSTM forecast (C). Each is evaluated on six demand scenarios with twenty
paired seeds.

## Contents

| Folder | Contents |
|---|---|
| `demand/` | Demand generator, partition and scenario builders, and their manifests |
| `network/`, `detectors/`, `config/` | Junction network, Webster signal programme and detector definitions |
| `scripts/` | Controllers, dataset extraction, model training, experiments, analyses and checks |
| `data/`, `data_broad/` | Training, validation and test datasets (original and broadened) |
| `models/`, `models_broad/` | The two trained models and their training reports |
| `results/` | Results of the three evaluations and the supplementary analyses |

Route files are not included: each is regenerated exactly from its seed by the partition
and scenario builders. Per-run SUMO output and working logs are also omitted.

Two sets of results, `results/experiment/` and `results/experiment_broad/`, were produced
before the timing fault described in Section 4.6.8 of the dissertation was corrected.
They are withdrawn and no reported result is taken from them.

## Environment

macOS on Apple silicon, Python 3.12.14 and SUMO 1.27.1 with `SUMO_HOME` set. The full
package list is in `requirements.txt`.

```
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 verify_environment.py      # expect 17 passes, 0 failures, 1 warning (libsumo)
```

## Reproducing the results

```
# network, partitions and fixed-time plan
bash network/build_network.sh && python3 scripts/make_tls.py
python3 demand/build_partitions.py
python3 demand/build_partitions.py --broad
python3 scripts/webster_calibrate.py --write

# datasets and models
python3 scripts/extract_dataset.py
python3 scripts/extract_dataset.py --partitions demand/partitions_broad --outdir data_broad
python3 scripts/train_lstm.py
python3 scripts/train_lstm.py --datadir data_broad --outdir models_broad --repeats 3

# scenarios and checks
python3 demand/build_scenarios.py --seeds 20
python3 scripts/conservation_checks.py
python3 scripts/test_detector_counting.py

# first comparison (original model) and remedial evaluation
python3 scripts/run_experiment.py --jobs 8 --outdir results/experiment_first_corrected
python3 scripts/analyse_experiment.py --outdir results/experiment_first_corrected
python3 scripts/run_experiment.py --jobs 8 --model-dir models_broad --outdir results/experiment_corrected
python3 scripts/analyse_experiment.py --outdir results/experiment_corrected

# confirmatory evaluation on the untouched scenario set
python3 demand/build_scenarios.py --seeds 20 --seed-base 90000 --outdir demand/scenarios_holdout
mv demand/scenarios demand/scenarios_comparison
cp -R demand/scenarios_holdout demand/scenarios
python3 scripts/run_experiment.py --jobs 8 --model-dir models_broad --outdir results/experiment_holdout
rm -rf demand/scenarios
mv demand/scenarios_comparison demand/scenarios
python3 scripts/analyse_experiment.py --outdir results/experiment_holdout

# supplementary analyses
python3 scripts/online_prediction_error.py
python3 scripts/detector_divergence.py --n-seeds 10 --out results/detector_divergence_corrected.json
```

`run_experiment.py` always reads `demand/scenarios`, so the confirmatory evaluation puts the
untouched set in its place for the run and restores the comparison set afterwards.

A rerun reproduces the traffic measures of the fixed-time and reactive controllers exactly.
`raw_results.json` will not match its checksum, because it also records wall-clock decision
latency, which varies between executions.

## Verifying the files

`CHECKSUMS.sha256` lists the SHA-256 checksum of every file in this repository, computed on
the machine that produced the results. To check them:

```
shasum -a 256 -c CHECKSUMS.sha256
```

## Notes on the code

Two comments are out of date and were left unchanged so that the checksums still identify
the code that produced the results. `demand/generate_demand.py` says the vehicle mix varies
with the profile, but the shares are fixed. `scripts/analyse_experiment.py` says the
practical thresholds were set in advance, but they were introduced during implementation.

`scripts/run_controller.py.prepatch` and `scripts/run_controller.py.prephase` are copies
saved before the two fixes described in Sections 4.6.4 and 4.6.8; neither was used for any
reported result.
