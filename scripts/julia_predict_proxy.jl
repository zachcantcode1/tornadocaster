#!/usr/bin/env julia

using JSON
import MemoryConstrainedTreeBoosting

"""
Proxy entrypoint for native Julia model inference.

Expected CLI:
  julia scripts/julia_predict_proxy.jl <request.json>

Request schema:
{
  "model_path": "...",
  "features": [[...], [...], ...],
  "output_path": "..."
}

Behavior:
  - Attempts native inference through MemoryConstrainedTreeBoosting.load_unbinned_predictor.
  - If dependency/import fails, falls back to zero predictions (integration-safe).
"""

function main()
    if length(ARGS) != 1
        println(stderr, "Usage: julia_predict_proxy.jl <request.json>")
        exit(1)
    end

    req_path = ARGS[1]
    req = JSON.parsefile(req_path)
    features = req["features"]
    output_path = req["output_path"]
    model_path = req["model_path"]

    n_rows = length(features)
    preds = zeros(Float32, n_rows)
    mode = "fallback_zeros"
    err = ""

    # Convert JSON nested arrays to Float32 matrix.
    # features shape from Python: (n_samples, n_features)
    X = Array{Float32}(undef, n_rows, n_rows == 0 ? 0 : length(features[1]))
    for i in 1:n_rows
        row = features[i]
        for j in 1:length(row)
            X[i, j] = Float32(row[j])
        end
    end

    try
        bin_splits, _trees = MemoryConstrainedTreeBoosting.load(model_path)
        required_features = length(bin_splits)
        current_features = size(X, 2)
        if current_features < required_features
            X = hcat(X, zeros(Float32, n_rows, required_features - current_features))
        elseif current_features > required_features
            X = X[:, 1:required_features]
        end

        predictor = MemoryConstrainedTreeBoosting.load_unbinned_predictor(model_path)

        # MemoryConstrainedTreeBoosting expects X as (samples, features).
        y = predictor(X)
        preds = Float32.(collect(y))
        mode = "native_mctb"
    catch e
        err = sprint(showerror, e)
    end

    open(output_path, "w") do io
        JSON.print(io, Dict("predictions" => preds, "mode" => mode, "error" => err))
    end
end

main()
