# Development Environment

Recorded on 2026-08-27 at 04:43 EDT for the MSc traffic-simulation research project.

## Host

| Item | Recorded value |
|---|---|
| Computer architecture | Apple Silicon (`arm64`) |
| Operating system | macOS 26.5.1, build 25F80 |
| Shell | `/bin/zsh` |
| Python installation method | Homebrew `python@3.12` |
| Python | 3.12.14 |
| Python executable | `/Users/opeyemiadeyeye/traffic_project/.venv/bin/python` |
| pip | 26.2.1 |
| Virtual environment | `/Users/opeyemiadeyeye/traffic_project/.venv` |

Homebrew's initial installation command returned a post-install warning after its auto-update process was blocked from writing lock files inside Homebrew's own Git repository. Python was nevertheless installed and linked. A direct test confirmed that Python 3.12.14 could create a virtual environment and that pip ran inside it, so the post-install warning was treated as cosmetic and was not retried.

Activate the environment with:

```bash
source ~/traffic_project/.venv/bin/activate
```

## Direct research packages

| Package/tool | Installed version | Notes |
|---|---:|---|
| Eclipse SUMO | 1.27.1 | Installed from the `eclipse-sumo` macOS arm64 wheel; the wheel also requires `sumo-data==1.27.1` |
| TraCI | 1.27.1 | Pinned to match SUMO |
| sumolib | 1.27.1 | Pinned to match SUMO |
| TensorFlow | 2.18.0 | Native macOS arm64 wheel |
| NumPy | 2.0.2 | Highest release in the NumPy 2.0 series and compatible with TensorFlow 2.18.0 |
| pandas | 3.0.0 | Requires NumPy >=1.26.0 on Python 3.12 |
| Matplotlib | 3.10.0 | Requires NumPy >=1.23 |
| SciPy | 1.18.1 | Resolver-selected; requires NumPy >=2.0.0,<2.8 |
| scikit-learn | 1.9.0 | Resolver-selected; requires NumPy >=1.24.1 |

`tensorflow-metal` was not installed. The research models use two small stacked LSTM layers, for which GPU transfer and dispatch overhead may outweigh computation time. CPU execution was retained as the simpler and more reproducible default. TensorFlow detected one physical and one logical CPU device and no GPU device.

XQuartz was not installed because the requested workflow is headless. The `sumo-gui` executable is present, but GUI use may require XQuartz and a logout/login cycle.

`libsumo` is not importable from this macOS installation (`No module named 'libsumo'`). High-volume runs must therefore use TraCI unless a separate, compatible libsumo distribution is deliberately added later.

## SUMO installation and shell environment

SUMO was installed inside the virtual environment with:

```bash
python -m pip install eclipse-sumo==1.27.1 traci==1.27.1 sumolib==1.27.1
```

The SUMO data root was discovered programmatically by inspecting the installed distributions and finding the common parent containing both `tools/` and `data/`:

```text
/Users/opeyemiadeyeye/traffic_project/.venv/lib/python3.12/site-packages/sumo
```

The following exact lines were added to `/Users/opeyemiadeyeye/.zshrc`:

```bash
# SUMO traffic simulation environment
export SUMO_HOME="/Users/opeyemiadeyeye/traffic_project/.venv/lib/python3.12/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools${PYTHONPATH:+:$PYTHONPATH}"
```

After opening a new shell, activate the virtual environment before running SUMO so its binary directory is on `PATH`.

## Deviations from the specified environment

### NumPy

- Requested: NumPy 2.4.0.
- Installed: NumPy 2.0.2.
- Reason: TensorFlow 2.18.0 declares `numpy<2.1.0,>=1.26.0`, so TensorFlow 2.18.0 and NumPy 2.4.0 cannot coexist in one valid environment.
- Decision: TensorFlow 2.18.0 was preserved because it governs model behaviour; the NumPy change is confined to array handling and does not affect results.

### SUMO

- Requested: SUMO 1.25.0.
- Installed: SUMO 1.27.1.
- Reason: The `eclipse-sumo==1.25.0` macOS arm64 wheel installed but its executables could not load their Homebrew-linked runtime dependency set. `sumo` first failed on Xerces-C and `netconvert` first failed on GDAL 36. The wheel also referenced Arrow/Parquet 1801, PROJ, FOX, Mesa, and X11-family libraries. Critical ABI revisions such as GDAL 36 and Arrow/Parquet 1801 are not what current Homebrew formulas provide.
- Decision: Building SUMO from source was rejected as disproportionate. Creating symlinks between incompatible library revisions was rejected as unsafe. The newest available macOS arm64 wheel, SUMO 1.27.1, executes cleanly. `traci` and `sumolib` were pinned to 1.27.1 to avoid an API mismatch.

### SUMO 1.25.0 wheel linkage audit

The table below distinguishes Mach-O ABI/library identifiers embedded in the 1.25.0 executables from current Homebrew *formula release versions*. These number systems are not interchangeable. The data was collected with `otool -L` on a separately downloaded 1.25.0 arm64 wheel and `brew info --json=v2` on 2026-08-27.

| Homebrew formula | Library identifier required by the 1.25.0 wheel | Mach-O recorded version | Current Homebrew formula |
|---|---|---:|---:|
| xerces-c | `libxerces-c-3.3.dylib` | 0.0.0 | 3.3.0 |
| gdal | `libgdal.36.dylib` | 36.3.10 | 3.13.3 |
| apache-arrow | `libarrow.1801.dylib`, `libparquet.1801.dylib` | 1801.0.0 | 25.0.1 |
| proj | `libproj.25.dylib` | 25.9.5 | 9.8.1 |
| gettext | `libintl.8.dylib` | 13.2.0 | 1.0 |
| fox | `libFOX-1.6.0.dylib` | 1.59.0 | 1.6.59 |
| libx11 | `libX11.6.dylib` | 11.0.0 | 1.8.13 |
| libxext | `libXext.6.dylib` | 11.0.0 | 1.3.7 |
| freetype | `libfreetype.6.dylib` | 27.2.0 | 2.14.3 |
| fontconfig | `libfontconfig.1.dylib` | 14.1.0 | 2.18.3 |
| libxft | `libXft.2.dylib` | 6.8.0 | 2.3.9 |
| libxcursor | `libXcursor.1.dylib` | 2.2.0 | 1.2.3 |
| libxrender | `libXrender.1.dylib` | 5.0.0 | 0.9.12 |
| libxrandr | `libXrandr.2.dylib` | 5.0.0 | 1.5.5 |
| libxfixes | `libXfixes.3.dylib` | 5.0.0 | 6.0.2 |
| libxi | `libXi.6.dylib` | 8.0.0 | 1.8.3 |
| jpeg-turbo | `libjpeg.8.dylib` | 8.3.2 | 3.2.0 |
| libpng | `libpng16.16.dylib` | 62.0.0 | 1.6.58 |
| libtiff | `libtiff.6.dylib` | 8.0.0 | 4.7.2 |
| mesa-glu | `libGLU.1.dylib` | 5.1.0 | 9.0.3 |
| mesa | `libGL.1.dylib` | 4.0.0 | 26.2.1 |

The 1.25.0 `sumo`, `netconvert`, and `duarouter` executables also reference macOS system libraries supplied by the operating system; those were present and are omitted from the table.

## Verification results

The reusable verification script is `/Users/opeyemiadeyeye/traffic_project/verify_environment.py`.

Run it with:

```bash
source ~/.zshrc
source ~/traffic_project/.venv/bin/activate
python ~/traffic_project/verify_environment.py
```

Result on 2026-08-27:

```text
PASS checks: 17
WARN checks: 1
FAIL checks: 0
WARN: libsumo is not importable: No module named 'libsumo'
OVERALL: PASS
```

Specific verified behaviour:

- Every direct package matched the version recorded above.
- `pip check` reported `No broken requirements found.`
- `SUMO_HOME` existed and contained `tools/` and `data/`.
- `import traci` and `import sumolib` succeeded.
- `sumo --version` and `netconvert --version` both executed and reported 1.27.1.
- A Mach-O linkage scan found no unresolved `/opt/homebrew` dependencies in the 1.27.1 `sumo`, `netconvert`, or `duarouter` binaries.
- TensorFlow 2.18.0 built and ran a forward pass through two stacked LSTM layers. The output shape was `(1, 1)`.
- TensorFlow detected `PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')` and `LogicalDevice(name='/device:CPU:0', device_type='CPU')`.

## End-to-end SUMO and TraCI smoke test

A throwaway three-node, two-edge network was generated with `netconvert`. Its connections input contained an explicit `linkIndex="0"` attribute, which `netconvert` accepted with exit status 0. One vehicle was then driven through both edges from Python via TraCI.

Observed vehicle counts were 1 on simulation steps 1 through 15 and 0 on step 16. TraCI closed normally. The complete throwaway directory was deleted after the successful test.

## Complete frozen Python environment

`/Users/opeyemiadeyeye/traffic_project/requirements.txt` was generated from `python -m pip freeze` after successful verification. Its complete contents are:

```text
absl-py==2.5.0
astunparse==1.6.3
certifi==2026.7.22
charset-normalizer==3.5.1
contourpy==1.3.3
cycler==0.12.1
eclipse-sumo==1.27.1
flatbuffers==25.12.19
fonttools==4.63.0
gast==0.7.0
google-pasta==0.2.0
grpcio==1.83.0
h5py==3.16.0
idna==3.19
joblib==1.5.3
keras==3.15.1
kiwisolver==1.5.0
libclang==18.1.1
Markdown==3.10.3
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.0
mdurl==0.1.2
ml-dtypes==0.4.1
namex==0.1.0
narwhals==2.25.0
numpy==2.0.2
opt_einsum==3.4.0
optree==0.20.0
packaging==26.3
pandas==3.0.0
pillow==12.3.0
protobuf==5.29.6
Pygments==2.21.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
requests==2.34.2
rich==15.0.0
scikit-learn==1.9.0
scipy==1.18.1
setuptools==84.0.0
six==1.17.0
sumo-data==1.27.1
sumolib==1.27.1
tensorboard==2.18.0
tensorboard-data-server==0.7.2
tensorflow==2.18.0
termcolor==3.3.0
threadpoolctl==3.6.0
traci==1.27.1
typing_extensions==4.16.0
urllib3==2.7.0
Werkzeug==3.1.8
wheel==0.48.0
wrapt==2.3.0
```
