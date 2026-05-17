# Project Tornadocaster: Agent Implementation & Testing Blueprint

## Context for the Implementing Agent
You are tasked with rebuilding and modernizing "Nadocast" (originally written in Ruby and Julia) into **Project Tornadocaster**. The goal is to create a Python-based, homelab-optimized, cloud-native microservice architecture. You will ignore any code related to Twitter/social media from the original repository and focus strictly on Data Ingestion, Feature Engineering, Inference, and a new LLM-based analysis layer.

**Reference Repository:** The original Nadocast source code can be found at [https://github.com/brianhempel/nadocast](https://github.com/brianhempel/nadocast). Refer to this repository whenever you need to understand the legacy mathematical formulas, model weighting, or ingestion schedules.

**Hardware Constraints:** This will run on lower-tier homelab hardware. You MUST strictly manage RAM (avoiding Out-Of-Memory crashes) and minimize disk I/O and network bandwidth.

**Core Stack:**
- **Language:** Python 3.11+
- **Data Processing:** `xarray`, `cfgrib`, `dask` (for chunked/tiled out-of-core operations)
- **Ingestion:** `httpx`, `asyncio`, `boto3` (for S3 and async HTTP byte-range requests)
- **Machine Learning:** `LightGBM` or `onnxruntime`
- **AI Integrations:** `openai` or `anthropic` client libraries 

---

## Step-by-Step Implementation Plan

### Step 1: The Ingestion Microservice (Zero-Footprint Pulls)
**Objective:** Fetch NOAA HREF/SREF/HRRR ensemble data without downloading full multi-gigabyte GRIB2 files.
- **Action:** Write Python async scripts (`httpx` or `aiohttp`) that target NOAA's AWS S3 Open Data Registry buckets (or Nomads `.idx` files). 
- **Mechanism:** Parse the `.idx` files to find the exact byte-range bounds for required atmospheric variables (CAPE, Shear, Dewpoint, Winds). Use HTTP `Range: bytes=start-end` headers to stream ONLY those exact binary chunks directly into memory.
- **Testing:** 
  1. Download a subset range.
  2. Decode the bytes in-memory using `cfgrib`.
  3. Assert that the spatial grid matches CONUS dimensions and contains expected float values (no NaN anomalies).

### Step 2: The Feature Engineering Pipeline (Tiled Processing)
**Objective:** Replicate the original Julia spatial calculations (means, gradients, 3-hour windows) and expand base variables into ~17,000 features without exceeding 16GB of RAM.
- **Action:** Re-write the math from `Grid130.jl` and `HREFPrediction.jl` using `xarray`.
- **Mechanism:** Use `xarray` rolling windows and convolutions for gradients and mean radii. CRITICAL: Use `Dask` to chunk the arrays spatially (e.g., cut the US map into 4 quadrants). Process one tile at a time to keep transient memory usage incredibly low.
- **Testing:** 
  1. Retrieve a historical weather date (e.g., April 27, 2011).
  2. Run the raw data through both the legacy Julia script and our new Python `xarray` pipeline.
  3. Use `numpy.testing.assert_allclose` to verify the generated feature arrays match to 4 decimal places.

### Step 3: ML Model Porting (Julia BSON to LightGBM/ONNX)
**Objective:** Port the pre-trained custom Julia memory-constrained decision trees to a standard modern framework.
- **Action:** Extract the tree weights, nodes, and thresholds.
- **Mechanism:** Write a one-off parser (in Julia or Python) to read the original `.bson` tree files. Map these into a standard JSON tree format compatible with `LightGBM` or convert them directly to `ONNX` architecture.
- **Testing:** 
  1. Create a mock array of 17,000 dummy feature values.
  2. Pass it into the legacy Julia `.bson` model.
  3. Pass it into the newly compiled `LightGBM`/`ONNX` Python model.
  4. Assert the final inferred probability floats are identical.

### Step 4: AI/LLM Integration (Automated Forecast Discussions)
**Objective:** Generate human-readable meteorological text explaining *why* the model predicted a threat.
- **Action:** Connect standard LLM APIs to the GBDT inference output.
- **Mechanism:** 
  1. After inference, extract the coordinates of the highest threat probabilities (`argmax`).
  2. Extract the "Feature Importances" from the LightGBM models for that specific geographic tile to determine *which* variables drove the high probability (e.g., SBCAPE, 0-1km SRH).
  3. Construct a strict system prompt: "Act as an SPC Meteorologist. Given the following coordinate anomalies and driving variables, write a mesoscale discussion explaining the threat."
- **Testing:** Shadow-run this component for 7 days. Generate internal text logs and manually review them against the official human-written SPC discussions to calibrate the prompt against hallucinations.

### Step 5: Orchestration (The Main Loop)
**Objective:** Tie the phases together into an automated daemon.
- **Action:** Write the central loop (`main.py`) that awaits the trigger, passes memory down the chain (using Apache Arrow if multi-processing is needed), and yields the final LLM text and probability coordinate maps.
- **Testing:** Run an end-to-end integration test mocking an S3 bucket upload trigger. Measure peak RAM usage (`memory_profiler`) to verify it stays below the targeted hardware constraints.
