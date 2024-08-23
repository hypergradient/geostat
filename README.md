# geostat

Model space-time data with Gaussian processes.

Geostat makes it easy to write Gaussian Process models with complex covariance
functions. It uses maximum likelihood to fit model parameters. Under the hood
it uses Tensorflow to fit models and do inference on GPUs. A top consumer GPU
such as an Nvidia RTX 4090 can handle 10k data points.

## Quickstart 

Install geostat using pip:
```
pip install geostat
```

## Examples

[here](doc/gaussian-processes-in-geostat.ipynb)

## Disclaimer

This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval by the U.S. Geological Survey (USGS). No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.

## Development and testing

pip install -e .[test]
