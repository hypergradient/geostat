# Welcome to GeoStat

Model space-time data with Gaussian processes.

Geostat makes it easy to write Gaussian Process (GP) models with complex covariance functions. It uses maximum likelihood to fit model parameters. Under the hood it uses Tensorflow to fit models and do inference on GPUs. A good consumer GPU such as an Nvidia RTX 4090 can handle 10k data points.

Visit our GitHub repository [here](https://github.com/whdc/geostat/tree/main).

---

## Quickstart

Install Geostat using pip:
```
pip install geostat
```

---

## Project layout

    README.md     # The readme file.
    mkdocs.yml    # The configuration file.
    doc/
    docs/
        about.md  # The about page.
        index.md  # The documentation homepage.
    src/geostat/
        __init__.py
        custom_op.py
        kernel.py
        krige.py
        mean.py
        mesh.py
        metric.py
        model.py
        op.py
        param.py
    tests/