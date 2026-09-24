# mobility_agent_ny — Manhattan replication

This is a fork of [kekehurry/mobility_agent](https://github.com/kekehurry/mobility_agent),
the code release for *"Graph RAG as Human Choice Model: Building a Data-Driven Mobility
Agent with Preference Chain"* (arXiv:2508.16172). The original paper only validates the
method on Cambridge, MA and San Francisco, CA. This fork replicates the same methodology
for **Manhattan, NY**, using real Replica, Advan (formerly SafeGraph), and Census ACS data,
to check whether the method's reported results generalize to a new, much larger and denser
city.

Git history is preserved from the original repo. Everything from here down documents what
changed and why, split into three parts: **bug fixes** (restoring behavior the paper itself
describes), **hardware-driven substitutions** (this ran on a laptop with a 4GB GPU, not
whatever the original authors used), and **methodological findings** (real limitations of
the method surfaced by testing it somewhere the original authors didn't).

## Bug fixes

Each is its own commit with the full rationale; summarized here for reference.

1. **`_similarity_search` was missing** (`mobility_agent/graph.py`) — called by
   `_get_reference_graph` but never defined anywhere in the released code, so any run with
   `use_reference=True` crashed. Implemented per the paper's sec 3.2 description (embed the
   query profile, rank all `person` nodes by cosine similarity, return the top-k).
2. **The graph's computed weights never reached the LLM** (`mobility_agent/agent.py`,
   `get_mode_prefernce`) — the paper (sec 3.4) describes the graph's probabilities as a
   *prior* the LLM calibrates; the original prompt listed the candidate mode/duration
   options with no numbers attached, so the LLM had nothing to calibrate against and just
   invented a fresh distribution. Now passes `prior_weight` per option explicitly.
3. **Profile-correction fallback crashed instead of degrading gracefully**
   (`mobility_agent/tools.py`, `ProfileGeneratorTool.generate_profile`) — added one retry
   (most failures here are a locally-hosted Ollama being briefly busy) and a counter of how
   many profiles fall back uncorrected; the fallback path itself also had two latent type
   bugs (age as a list, not int; `available_vehicles` as a category string, not int) that
   crashed instead of returning a usable profile.
4. **`sentence_transformers` import was fragile and unused everywhere but one call**
   (`mobility_agent/graph.py`) — pulled in the full torch/transformers stack for a single
   `util.cos_sim()` call, when the class already has its own numpy `_cosine_similarity`. The
   import was also actively broken on a fresh install (needs torch>=2.5, this environment
   has 2.4.1). Removed the dependency entirely.
5. **`geopandas.read_file()` crashed on non-cp1252 characters** (`mobility_agent/tools.py`)
   — GDAL's CSV driver defaults to the system codepage on Windows; Manhattan POI names with
   characters outside cp1252 (from the source Advan data) crashed outright instead of just
   rendering oddly. Passes `encoding='utf-8'` explicitly.
6. **LLM output parsing wasn't robust to minor format drift** (`mobility_agent/agent.py`,
   `mobility_agent/tools.py`) — a strict `strptime("%H:%M")` on the LLM-generated schedule's
   start time crashed (and silently dropped that whole activity) on anything but an exact
   match, e.g. added seconds; a trailing/double comma in the LLM's NAICS code list crashed
   `int('')`. Smaller local models (see below) hit these far more often than whatever model
   the original authors used. Time parsing now falls back to `dateutil.parser`; NAICS
   parsing filters empty pieces.
7. **Desire-weighting embedded one node at a time instead of batching**
   (`mobility_agent/graph.py`, `_get_reference_graph`) — up to k=20+ sequential HTTP calls
   to the local embedding server per single `preference_modelling()` call, each one a
   chance to stall. Batched into one call, matching how `_embedding_graph` already batches
   person-node embeddings.

## Hardware-driven substitutions

This ran on a laptop with an **NVIDIA RTX 2050 (4GB VRAM)** — the paper uses `qwen3:8b`
(~5.2GB quantized). None of this is a methodology change; it's what was necessary to get a
working local LLM at all without paid API access.

- **Model: `qwen2.5:3b` instead of `qwen3:8b`.** `qwen3:8b` and even `qwen3:4b` didn't fit
  in 4GB alongside the embedding model (`mxbai-embed-large`) and got partially offloaded to
  CPU — a single full agent took 2+ hours, sometimes longer. `qwen3`'s `/no_think` workaround
  (needed because it's a hybrid reasoning model) also never reliably suppressed its
  "thinking" step here, adding hundreds of wasted tokens per call. `qwen2.5:3b` has no
  reasoning mode to fight with, fits fully in VRAM, and measured ~35-41 tok/s. `llama3.2:3b`
  was also tried (the paper's own Appendix found Llama competitive with Qwen3 in some
  conditions) but was both slower end-to-end and hit more distinct parsing failures in
  practice — see `Modelfile.qwen2.5-3b` / `Modelfile.llama3.2-3b` for the exact configs
  tried, including a reduced `num_ctx` (2048, down from the 4096 default) needed to keep
  the LLM fully GPU-resident alongside the embedding model.
- **Reduced reference graph size for exploration (`sample_num=5000`) vs. actual experiments
  (`sample_num=50`).** The paper itself found 50-100 reference samples sufficient (diminishing
  returns beyond that) and used 50 for its own full-population experiment (sec 5) — the 5000
  figure was only used early on to explore the graph's structure, not for the numbers below.
- A large chunk of debugging time went to **orphaned `llama-server.exe` processes**
  accumulating VRAM usage across repeated kills/restarts during iteration — not a code issue,
  but worth knowing if this pipeline mysteriously slows down again: check `nvidia-smi` for
  more `llama-server.exe` processes than loaded models, and kill everything before restarting
  Ollama.

## Methodological findings (not fixed — documented)

These are real properties of the method/paper that testing on Manhattan surfaced. Fixing
them would change the methodology being reproduced, which isn't this fork's goal.

- **Replica's "Home Location" filter conditions on having taken a trip.** To restrict a
  Replica study area's People export to Manhattan, you filter by Home Location — but per
  Replica's own documentation, once *any* filter is active the People table represents only
  "the traveling population," not full residents. There's no way in the Studio UI to get a
  geographically-restricted, non-traveler-inclusive population, so `demographics.json` is
  built from a population implicitly conditioned on having traveled that day.
- **The reference graph's row-level sampling means "reference schedules" aren't anyone's
  actual day.** `BehaviorGraph._load_data()` samples individual trip *rows*, not people —
  each reference person contributes ~1 of their ~4 daily trips on average. When
  `make_a_plan()` builds a "reference schedule from people with similar profile" to show the
  LLM, it's actually a collage of single moments from many different people, not one
  person's coherent day. (A person-level sampling variant — same person count, but pulling
  *all* their trips — would fix this and is a natural follow-up experiment, not attempted here.)
- **POI search's "20 nearest" cutoff neutralizes the point of a wide search radius.**
  `POISearchTool.find_nearby_pois` filters by the LLM-chosen `[min, max]` distance, then
  always truncates to the 20 physically closest results before the LLM ever sees them. In a
  POI-dense area like Manhattan (e.g. 11,047 restaurants city-wide), a wide radius (meant to
  let an agent traveling by car go further) almost never matters in practice — proximity
  wins regardless, except for genuinely rare categories.
- **`desire`/schedule-activity type is a free-form string, not an enum**, despite being
  described as one of 9 fixed categories (`ScheduleItem.desire: str` in `agent.py`) — the
  LLM isn't constrained to emit exactly one of them. The graph's downstream desire-matching
  step uses semantic similarity (rather than exact match) partly to tolerate this, and partly
  because a small retrieved neighborhood (k~20 people) may not contain an exact category
  match at all — exact-match would sometimes make every `want_to` edge weight 0 and crash the
  normalization step.
- **The duration schema implicitly assumes no trip exceeds 60 minutes.** The paper's own
  data schema (Table 1) and the LLM prompt's `Available Duration Choices` both stop at
  `50-60`. 11.6% of the raw Manhattan eval set exceeds that (max 1415 minutes) — likely
  because Manhattan's real commute-shed is much larger than Cambridge/SF's. Since
  `get_mode_prefernce` returns mode and duration as one coupled `(mode, duration)` choice
  (not decided independently — see the graph's merged `Intention` nodes and the
  `choice_weights` prompt format), a duration outside the offered range could also distort
  the *mode* prediction for that row, not just duration. `scripts/build_manhattan_eval_max60.py`
  builds a fresh 1000-row eval set staying within the schema's implicit assumption, rather
  than silently keeping the out-of-range rows in the reported numbers.

## Data pipeline (Manhattan)

All under `data/`, built by the scripts in `scripts/`:

| File | Source | Script |
|---|---|---|
| `reference/replica-manhattan_trips.csv` | Replica trips export, 5000-row sample | `build_manhattan_ref_eval_split.py` |
| `eval/replica-manhattan_trips_eval_max60_1000.csv` | Replica trips export, 1000-row sample, duration<=60min | `build_manhattan_eval_max60.py` |
| `geo/replica-manhattan-cencus.geojson` | NYC Open Data (tract polygons) + Census ACS 5yr (real population per tract) | `build_manhattan_census.py` |
| `geo/replica-manhattan-demographics.json` | Replica People export, 888,862 unique people | `build_manhattan_demographics.py` |
| `geo/replica-manhattan-poi.csv` | Advan Weekly Patterns Plus, 4 weeks, 38,203 unique Manhattan POIs | `build_manhattan_poi.py` |

`CENSUS_API_KEY` in `.env` is a free key from the Census Bureau
(https://api.census.gov/data/key_signup.html), used only by `build_manhattan_census.py`.

## Current status

- Experiment I (mode/duration prediction against real eval trips, matching
  `exp1_step1_agent.ipynb`) — see `scripts/run_experiment1_manhattan.py`. A 1000-row run
  is the target; smaller test runs (20, 100 rows) came back clean (no errors) after the
  fixes above, with mode exact-match accuracy around 20-36% (vs. ~14% random baseline
  across 7 mode categories) — a rough sanity signal, not the paper's actual KLD/MAE metric,
  which still needs to be computed with `baseline/eval.py`.
- Experiment II (full synthetic population with day-long schedules, matching
  `exp2_step1_abm.ipynb`) — attempted at small scale (1-2 agents) during development;
  works end-to-end but slowly (minutes per agent) and with occasional dropped activities
  from the parsing-robustness issues above. Not run at the paper's scale (1000 agents) —
  see the hardware section above for why that isn't practical on this machine.

---

# Original setup instructions (upstream)

The sections below are from the original repo. Where this fork's setup differs (Manhattan
data files, `qwen2.5:3b` instead of `qwen3-nothink:8b`), see `.env` and the Modelfiles in
this directory instead of the commands below, which are kept for reference/comparison.

## Init Environment

### Create and activate a virtual environment
```
uv venv
source .venv/bin/activate
```

### Install dependencies from pyproject.toml
```
uv sync
```

## Download POI Data

```
gdown 1ZMA_tNP6EapE6UED8hyi5_6uoNzTSvt0 -O data/geo/safegraph-sf-poi.csv
gdown 1JnOaM22HSrWC8SWVKm2x7kJHjl0oHoVt -O data/geo/safegraph-cambridge-poi.csv
```

## Create Qwen3-nothink Model

The experiments are using `qwen3:8b`, and the thinking mode is off by default. Using the `Modelfile` to create this model

```
ollama create qwen3-nothink:8b -f Modelfile
```

# Example Usage

## Mode Preference Modelling

```python
from mobility_agent.graph import BehaviorGraph
graph = BehaviorGraph()

profile = "a young women"
desire = "eat"
time = 8
subgraph,weights = graph.preference_modelling(profile=profile,desire=desire,time=time,k=10,depth=4)
graph.visualize_graph(subgraph,node_size=600,font_size=16,title_size=16)
print("Transportation Mode Preference:",weights)
```

## Agent-based Modelling

### Generate Profile

- The agent generate realistic profiles based on demongraphic data
- It then choose a random location as home wighted by cencus block population

```python
from mobility_agent.agent import MobilityAgent

agent = MobilityAgent(city="Cambridge,MA")
print("Profile:",agent.profile)
print("Home:", agent.memory_locations)
```

### Create Time Schedule

- The agent first creates a full-day activity plan.

- For each activity, it searches for nearby points of interest (POIs) based on their profile, and prefered mode of transportation.

- It then selects the most suitable POI to visit and save it to the working meomory of the agent.

- The agent will save important locations such as home, work, school to `agent.memory_locations`

```python
agent.get_time_schedule()
print("Time Schudule:",agent.time_schedule)
```

### Memories


```
print(agent.working_memory)
```


```
print(agent.memory_locations)
```
