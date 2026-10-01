# RVC server runtime with uv

## Status and scope

Migration began on 2026-09-30. **The permanent uv environment is installed and
the isolated comparisons pass.** RVC 1.1.0 and CLI 3.16.0 are released, and
the operator confirmed a successful uv launch with the RTX 4090 on port 7861.
On 2026-10-01 UTC, the owner waived further manual conversion/training checks:
there is no planned use or prepared test workflow, and any future issues will
be addressed when the application is used. Further manual acceptance is not an
outstanding migration step. Keep the original Conda environment for rollback
until retirement is separately approved.

This checkout is `/srv/farm/code/rvc`, the manually launched RVC WebUI and
training/export tools. No installed standalone RVC systemd service was found in
the migration audit. **No service restart or privileged installation command is
required for this checkout.** Finish any standalone RVC work before switching
its next launch to another interpreter.

Production Morph uses the API's own RVC package and Python environment. Migrating
this checkout does not migrate the API/Morph runtime. The CLI's `farm download
rvc` command also uses the API environment and remains unchanged. No API worker
restart belongs to this migration.

## One environment layout

- Environment: `/srv/farm/code/rvc/.venv`
- Managed Python: `/srv/farm/.uv/python/cpython-3.11.14-linux-x86_64-gnu`
- Shared hardlinked package cache: `/srv/farm/.uv/cache`
- Baseline and dependency provenance: `/srv/farm/.uv/migrations/2026-09-30-rvc`
- Rollback interpreter: `/home/gradywoodruff/miniconda3/envs/rvc/bin/python`

The lock targets Farm's Linux x86_64 server. Create the environment at its final
path; do not copy it between repositories or machines. This follows the
[Farm runtime layout](/srv/farm/docs/guides/runtime-environments.md), with no
central uv environment directory. Python remains **3.11.14**, not the 3.11.15
used by some other projects: each environment preserves its own baseline.

`pyproject.toml` and `uv.lock` preserve **all 150 installed package versions** from
the original Conda environment, including packaging/development tools. The
baseline includes Torch and torchaudio 2.5.1+cu121, torchvision 0.20.1+cu121,
NumPy 2.3.5, Gradio 3.34.0 and Fairseq 0.12.3. This is not a dependency upgrade or
pruning pass. Matching version numbers alone does not prove native builds or
results are identical; the completed runtime comparisons are recorded below.

The previous Poetry manifest is preserved at
[docs/history/pyproject-poetry.toml](history/pyproject-poetry.toml). It did not
reproduce the installed Farm environment. Upstream platform/requirements
instructions elsewhere in the repository are reference material; use this guide
for the Farm server. Do not run Poetry or ad hoc `pip install` into this `.venv`.

## Dependency provenance

**Fairseq is the installed fork, not PyPI Fairseq.** Its source is
`https://github.com/One-sixth/fairseq.git`, pinned to commit
`44800430a728c2216fd1cf1e8daa672f50dfacba`. Isolated build dependencies are
constrained to the baseline, including NumPy 2.3.5, Cython 3.2.4,
setuptools 80.10.2 and wheel 0.46.3. Torch has an explicit
`torch==2.5.1+cu121` build constraint, and
`tool.uv.extra-build-dependencies.fairseq` supplies the exact CUDA 12.1 wheel URL
and SHA-256 as an actual build requirement. Keep both: with uv 0.8.19, a URL in
a build constraint alone did not select that artifact. The build must use the
same Torch variant as the runtime; see [uv's build-dependency guidance](https://docs.astral.sh/uv/concepts/projects/config/#augmenting-build-dependencies).

When correcting build dependencies, remove only the migration-generated Fairseq
build tree and invalidate only Fairseq's package cache before rebuilding:
setuptools can otherwise reuse compiled objects from the previous Torch ABI.
This migration performed that targeted clean rebuild. Its
[build provenance](/srv/farm/.uv/migrations/2026-09-30-rvc/dependency-provenance/fairseq-build/)
is retained separately from the application source. All **six** native extensions
import successfully in the clean uv build, with no Conda library mappings. The
original Conda environment already failed to import `libbase` and
`alignment_train_cpu_binding`; the clean build also resolves that existing ABI
problem. The migration preserves the Fairseq source revision and application
code.

A first installation can require source compilation. The runtime helper limits
concurrent builds to one and defaults `MAX_JOBS` to two to bound build load on
the shared server.

**ONNX Runtime GPU 1.24.2 uses its original PyPI wheel.** Its release metadata was
unavailable, so the original wheel URL and digest were recovered from the cached
PyPI index entry. `tool.uv.sources` selects that exact `files.pythonhosted.org`
artifact; `uv.lock` records this SHA-256:

```text
88ed80d143dfba314f1951305079665e6e86516bf19d96dfaf8c0f56cd9fae1b
```

The recovered index entry and wheel provenance are retained under
`dependency-provenance/onnxruntime-gpu-1.24.2/` in the migration record. A missing
release listing is not permission to substitute a different ONNX version.

**grpcio 1.78.1 remains an explicit baseline exception.** uv reports that this
release was yanked, citing upstream problems in Google Cloud serverless
environments. This migration preserves the working installed version; it does
not declare the package generally safe or silently upgrade it. Any later package
alignment needs its own validation. Both environments' `pip check` reports no broken requirements.

## Install, check and release

The environment is already installed. The supported reproduction/update commands are:

```sh
cd /srv/farm/code/rvc
sh scripts/runtime.sh sync
sh scripts/runtime.sh check
```

`sync` installs the pinned managed Python if needed, then runs `uv sync --locked`
into `.venv`. It selects the shared Farm Python/cache directories explicitly.
The default uv executable is `/home/gradywoodruff/.local/bin/uv`; `UV_BIN` is an
explicit override. The helper does not upgrade uv or install OS packages.

`scripts/python-downloads.json` supplies the pinned Astral Python 3.11.14 Linux
GNU x86_64 build `20260211` and its SHA-256 because the installed uv download
catalog predates it. `uv python install --no-bin` does not change the system
Python or the user's default Python. Host `ffmpeg`/`ffprobe` commands remain
external requirements.

`check` verifies the lock offline without synchronizing installed packages. It
is also the `farm.json` `release.commands` hook. The non-installed runtime
metadata has version `0.0.0`; the application's release version remains in
`farm.json`, so release version changes do not require a lockfile version rewrite.
The hook does not install packages, launch RVC, download models or restart
anything. The old `venv.sh` is now a compatibility entry point to the same helper:
`sh venv.sh` synchronizes; `sh venv.sh check` checks the lock.

The user runs normal releases: release RVC first, then the CLI. Synchronize after
an accepted release that changes dependencies, with standalone RVC stopped if
installed packages will change. Never modify shared hardlinked package files in
place. A dependency change should update declarations and pins together, resolve
and review the lock, and repeat the relevant comparisons.

## Launch and export

```sh
cd /srv/farm/code/rvc
./run.sh
```

The Farm launcher selects `.venv/bin/python`, starts `infer-web.py` on port
**7861**, and passes `--noautoopen`. It preserves the existing WebUI bind address
`0.0.0.0`. Additional arguments are forwarded, for example `./run.sh --port 7862`.
`farm app rvc` and its aliases use this same script through
`/srv/farm/sys/cli/commands.server.json`.

Launch is separate from installation. `run.sh` fails with setup instructions if
the interpreter is missing; it never creates an environment, invokes sudo,
installs packages or downloads pretrained models. Existing weights, training
experiments and assets stay in place. The obsolete automatic Python 3.8 setup
and model-download path has been removed from this launcher.

The WebUI's `Config` defaults child preprocessing, feature extraction and
training commands to `sys.executable`, so they follow the selected interpreter.
An explicit `--pycmd` remains an upstream override. The launcher first changes
to the repository directory because the WebUI resolves configs/assets/logs
relative to the working directory.

The repository's existing export script now uses the same project interpreter:

```sh
cd /srv/farm/code/rvc
farm run export --help
```

This runs `/srv/farm/code/rvc/.venv/bin/python tools/export_model.py`; pass the
normal export arguments to perform an export. No model export is required just
to migrate the runtime.

## Validation and acceptance

Completed checks at the permanent environment path:

| Check | Result |
| --- | --- |
| Python and packages | Python 3.11.14; all 150 distribution versions match; clean dependency check |
| Runtime independence | Managed Python, no Conda native mappings, cache hardlink sharing verified |
| Locked recreation | Offline repeated sync audits all 150 packages without changes; release lock check passes |
| UI | Real temporary loopback server returns HTTP 200 and 199 Gradio components |
| Preprocess, RMVPE pitch, HuBERT features | Exact baseline outputs |
| PyWorld native extension | Harvest and stonemask produce exact baseline arrays |
| CPU conversion | Two seeded repeats and cross-environment PCM/WAV hashes match exactly |
| GPU conversion | Five runs per environment; all 25 cross-pairs pass predeclared bounds |
| Training and resume | Actual FP16 optimizer updates from normal pretrained weights; finite generator/discriminator checkpoints; resume to epoch 2 |
| Export | Latest resumed model and FAISS index copied with matching hashes |
| Native Fairseq extensions | All six load against the matching Torch build |
| Launchers and CLI | Interpreter/cwd/arguments/rollback tests plus all 354 CLI tests pass |

Native GPU math varies slightly even within the original environment. Bounds
were frozen from five Conda runs before any uv GPU outputs: RMS difference at
most 7.17234 PCM16 units, maximum difference 112, correlation at least 0.99999,
and matching sample rate/shape. Final worst cross-pair RMS was
3.58617, maximum 56, and minimum correlation
0.999999647. CPU comparisons are exact. No model precision or
engine setting was changed for native GPU acceptance.

The training smoke test uses eight repeated copies of one short sample and the
normal UI pretrained generator/discriminator. It exercises optimizer updates,
checkpointing and resume; it is not a quality assessment of a fully trained
voice. Initial randomly initialized FP16 probes skipped optimizer updates in the
original environment, and an attempted deterministic GPU probe hit an existing
unsupported deterministic kernel. Those baseline limitations are preserved in
the evidence and were not treated as successful validation.

See [comparison.json](/srv/farm/.uv/migrations/2026-09-30-rvc/validation/comparison.json),
[runtime-validation.json](/srv/farm/.uv/migrations/2026-09-30-rvc/runtime-validation.json),
and the validation evidence manifest for inputs, scripts, logs and exact hashes.
Tracked application source and the voice input retain their captured baseline
hashes. Model/index assets were only read through individual links; their old
timestamps remain intact and current hashes are recorded, but pre-run hashes
were not captured for those large files. The temporary validation UI was
closed after testing. The later operator launch succeeded; manual conversion
and training acceptance were waived by the owner, not performed. This decision
is recorded in
[manual-acceptance.json](/srv/farm/.uv/migrations/2026-09-30-rvc/manual-acceptance.json).

Validation must use isolated source/config/log/model copies and disposable
outputs. Importing `infer-web.py` has startup side effects: it recreates `TEMP`,
creates directories and loads dotenv; `Config` may copy/rewrite `configs/inuse`
for the selected device/precision. Do not import the live checkout merely to
inspect arguments or run CPU smoke checks there. Preserve production model and
source hashes across validation.

Bounded GPU comparisons use a temporary durable reservation of Farm's shared
GPU key, one stage at a time, without creating queue tasks. The wrapper refuses
active resource allocations or another owner, verifies resident service process
identities and idle GPU samples, and requires Color retraining to be inactive.
It heartbeats only its own ownership and terminates the complete owned process
group before releasing. A busy response means wait; never steal another lock or
remove a durable lock based only on age. An abruptly killed wrapper can leave a
reservation intentionally; investigate its recorded owner and descendants before
any manual recovery. Resident services are not stopped for validation.

The owner has closed further manual testing for this migration. Conda remains
available for rollback; removing it is a separate retirement decision. No net
disk saving is claimed while the rollback environment and migration artifacts
remain installed.

## Rollback

Stop only the standalone RVC process you launched, after its current work has
finished, then select the original interpreter explicitly:

```sh
cd /srv/farm/code/rvc
FARM_RVC_PYTHON=/home/gradywoodruff/miniconda3/envs/rvc/bin/python ./run.sh
```

The override also carries through the WebUI's default child Python selection.
`farm run export` uses its declared `.venv` path, so export rollback is a direct
invocation of the retained interpreter:

```sh
cd /srv/farm/code/rvc
/home/gradywoodruff/miniconda3/envs/rvc/bin/python tools/export_model.py --help
```

Replace `--help` with the normal export arguments when needed. Do not restore old
model/log/config snapshots over newer user work to undo an interpreter change.
Conda retirement requires a separate consumer/process audit and an explicit
retirement decision. No service restart or privileged command is needed for
this manual-launch rollback.
