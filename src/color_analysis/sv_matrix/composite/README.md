# Combine two single-color models into a composite model
- Uses the same `training_split` as the individual color models
- Generates the `util_cdf.txt` file for the composite model
- The flag `--composite-or`, if set, indicates that the composition operator is OR. Otherwise it is AND.
- This script generates the composite model files that can be used by the `sv_matrix_model` in the `python_server`.
`python3 compose_single_color_models.py -C <color-1-model> <color-2-model> -V <color-1-vid-dir> <color-2-vid-dir> [--composite-or] -O <outdir-for-composite-model> -T <training-split>`

# Perform cross-validation
NOTE: Right now it is hardcoded to work for OR composition only.
`./cross_validation.sh <train-conf-color-1> <train-conf-color-2> [<--composite-or>] <dump-dir>`

# Plotting ONLY for composite utility 
`python3 compute_composite_utils.py -C <color-1-train-conf> <color-2-train-conf> -O <outdir> [--composite-or]`

# Plotting per-color utility threshold and observed drop ratio vs target drop ratio
`python3 compute_single_video_multi_color_thresholds.py -U <color-1-frame-utils.csv> <color-2-frame-utils.csv> -C <color-1> <color-2> -R <training_split>`
