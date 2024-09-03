# Geostat

Model space-time data with Gaussian processes.

Geostat makes it easy to write Gaussian Process (GP) models with complex covariance
functions. It uses maximum likelihood to fit model parameters. Under the hood
it uses Tensorflow to fit models and do inference on GPUs. A good consumer GPU
such as an Nvidia RTX 4090 can handle 10k data points.

## Quickstart 

Install Geostat using pip:
```
pip install geostat
```

## Examples notebooks

  * [An introduction to Geostat.](docs/notebooks/gaussian-processes-in-geostat/gaussian-processes-in-geostat.ipynb)
  In Geostat, we create one model that is used to create synthetic data according
  to provided parameters, and we create a second model that does the inverse:
  it takes the data and infers the parameters.
  * [Structured covariance functions.](docs/notebooks/3d-gaussian-processes/3d-gaussian-processes.ipynb) Here
  we show how a progressively more complex covariance function fits data better
  than simpler ones.
  * [Making predictions in a shape.](docs/notebooks/predictions-with-mesh/predictions-with-mesh.ipynb) Geostat
  has utility functions to make it easier to work with shapes.
  * [Gaussian processes in Tensorflow.](docs/notebooks/gaussian-processes-in-tensorflow/gaussian-processes-in-tensorflow.ipynb) A tutorial on how to implement Gaussian processes in Tensorflow.

## Documentation

Coming soon.

## Disclaimer

This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval by the U.S. Geological Survey (USGS). No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.
