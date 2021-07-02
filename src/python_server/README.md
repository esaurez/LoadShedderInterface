The `example_calls.py` file can be used to test the implementation of the server (esp. the domain specific calls to compute util threshold and util).

# Build the Model
To build the model, you should call the script [request_handler.py](request_handler.py) as follows:
```sh
python request_handler.py config.properties
```

- The file [config.properties](config.properties) contains five parameters which are already explained in [config.properties](config.properties).

- [request_handler.py](request_handler.py) script loads all '\*.bin' files from trainingDataPath directory. trainingDataPath is defined in [config.properties](config.properties).

- [request_handler.py](request_handler.py) calls [build_model.py](../model/build_model.py) to build the model.

- The trained model is then stored in a directory specified by generatedModelPath parameter in [config.properties](config.properties).

- Note: To use different bin sizes for the image histograms, you should change fullHistogramBinSize parameter in [config.properties](config.properties). For each different bin size, you can use a different output path, i.e., generatedModelPath, in [config.properties](config.properties) to store the trained models for different bin sizes.

# Test the model
To test the trained model, you may use [test_shedding.py](test_shedding.py) script. Use the following command to call the script.
```sh
python test_shedding.py -p config.properties -d drop_ratio -r path/to/result/directory
```
- The script uses again variables from the properties file [config.properties](config.properties).

  - It uses generatedModelPath parameter to find and load the trained model.

  - Then, it gets the utility threshold for the given drop ratio.

  - It uses trainingDataPath to  find all '\*.bin' where it loads all frames from all these  '\*.bin' files, predicts the utility for the frames using the loaded trained model,  and decides whether or not to shed frames depending on their utilities and the computed utility threshold.

  - It saves the shedding results in path/to/result/directory using the csv file format.

  - Note: To test a trained model with each '\*.bin' individually, you may create folders with single '\*.bin' files. And, in every call of [test_shedding.py](test_shedding.py), you change trainingDataPath in [config.properties](config.properties) to point out to these folders. In this case, you should make sure to change  path/to/result/directory to change the output '\*.csv' file name to not override the shedding results for previous '\*.bin' files.

# Plot the Results
To plot the results, you may use [compute_false_positives_and_negatives.py](../plot_results/compute_false_positives_and_negatives.py) script. Use the following command to call the script.
```sh
python compute_false_positives_and_negatives.py -r path/to/result/generated/from/shedding
```

- The script reads all '\*.csv' files from the folder path/to/result/generated/from/shedding. It  computes the percentage of false negatives for each '\*.csv' file and plots the results in a single figure using bar plot. It uses '\*.csv' file names as the ticks for the x-axis.

- Note: Plotting the results for different bin sizes or when testing the shedding impact on each '\*.bin' file individually. Just place all generated load shedding result files,i.e., '\*.csv' files, in a single directory (if they are not already in a single directory). Then, call  [compute_false_positives_and_negatives.py](../plot_results/compute_false_positives_and_negatives.py) to plot the results.
