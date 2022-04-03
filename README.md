
# Readme

## Library for tracing and inferencing (/library)
### Dependencies

`spack env activate .`

`spack install`

### Build

`meson bld && ninja -C bld`

### Configuration

| Option                        | Description                                      |   Mode   | Inferencing mode |
|-------------------------------|--------------------------------------------------|:--------:|:----------------:|
|                               |                                                  | Sampling |    Inferencing   |
| -m, --min-size=9              | Min size of chunks to analyze in bytes           |     X    |         X        |
| -r, --repeat=3                | Number of times to repeat measurements           |     X    |                  |
| -p, --meta-path=/tmp/meta.h5  | Path for metadata storage                        |     X    |         X        |
| -t, --tracing                 | Activates tracing of MPI-Calls                   |     X    |                  |
| -s, --store-chunks            | Activates chunk storage                          |     X    |                  |
| -c, --chunk-path=/tmp/chunks/ | Storage path of chunks for later analysis        |     X    |                  |
| -e, --test-compression        | Activates compression tests according to metrics |     X    |                  |
| -x, --model-path              | Path to exported ONNX model                      |          |         X        |
| -o, --settings-path           | Path to exported ONNX settings                   |          |         X        |
| -i, --inferencing             | Run inferencing                                  |          |         X        |
| -d, --decompression           | Measure decompression                            |     X    |                  |


### Usage example
`export IOA_OPTIONS="--repeat=3 --tracing --decompression --test-compression --meta-path=meta.h5 --chunk-path=chunks/`
`G_MESSAGES_DEBUG=all LD_PRELOAD=bld/libmpi-preload.so mpiexec -np 2 application`

### Usage inferencing
 Specify model and model settings files used in training step
 
`export IOA_OPTIONS="--min-size=9 --meta-path=evaluation.h5 --inferencing --model-path=compression-CR.onnx --settings-path=compression-CR-settings.txt`

# Training and evaluation (/CompressionML-PyTorch)
## Dependencies
- Uses [Poetry](https://python-poetry.org/docs/basic-usage/) for dependency management

`poetry shell && poetry install`

## Usage
Allows for hyperparameter tuning, as well as final model creation with the discovered parameters.

### `tuning.py`
- Specify meta.h5 and metric within file
- Run file and discover parameters, e.g by using [Tensorboard](https://www.tensorflow.org/tensorboard)

### `training.ipynb`
- Use discovered parameters from previous step and train final model

## Evaluate
### `confusion.ipynb`
- Set meta path to *evaluation.h5* output path specified in `IOA_OPTIONS` when inferencing
