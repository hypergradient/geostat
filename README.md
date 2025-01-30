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

  * [An introduction to Geostat.](doc/gaussian-processes-in-geostat.ipynb)
  In Geostat, we create one model that is used to create synthetic data according
  to provided parameters, and we create a second model that does the inverse:
  it takes the data and infers the parameters.
  * [Structured covariance functions.](doc/3d-gaussian-processes.ipynb) Here
  we show how a progressively more complex covariance function fits data better
  than simpler ones.
  * [Making predictions in a shape.](doc/predictions-with-mesh.ipynb) Geostat
  has utility functions to make it easier to work with shapes.
