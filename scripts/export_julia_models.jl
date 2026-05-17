using Pkg
# Pkg.add("BSON")
# Pkg.add("LightGBM")
using BSON

# This script loads an original Nadocast .bson file and extracts the LightGBM booster string or saves it to a .txt file.

function export_lightgbm_bson(bson_filepath::String, output_txt_path::String)
    println("Loading $bson_filepath...")
    data = BSON.load(bson_filepath)
    
    # Typically, the LightGBM model is stored under a key in the BSON dict, or it's just the object itself.
    # We will inspect the keys.
    println("Keys in BSON: ", keys(data))
    
    # Extract the booster / string. 
    # (Adjust 'model' key if Nadocast used a different struct field)
    model = haskey(data, :model) ? data[:model] : data
    
    # Check if we can save it natively using LightGBM or if it's already a string array
    if hasproperty(model, :model_str)
        open(output_txt_path, "w") do f
            write(f, model.model_str)
        end
        println("Exported model_str directly to $output_txt_path")
    else
        # Fallback: assuming LightGBM.jl booster
        # LightGBM.savemodel(model, output_txt_path)
        println("Need explicit LightGBM.savemodel definition for this object type: ", typeof(model))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        println("Usage: julia export_julia_models.jl <input.bson> <output_model.txt>")
        exit(1)
    end
    export_lightgbm_bson(ARGS[1], ARGS[2])
end
