## Development and testing

```
pip install -e .[test]
```

## Build

To build:
```
python -m build
```
To upload to PyPi:
```
twine upload --repository geostat
```

## Documentation

The documentation is hosted on GitHub pages on this [link](https://hypergradient.github.io/geostat/).
The source folder of the documentation website is the gh-pages branch. Unfortunately, the documentation does not allow for automatic updates yet.

1 - Install mkdocs
```
pip install mkdocs
pip install mkdocs-material
pip install 'mkdocstrings[python]'
```

2 - Navigate to the GitHub repository and checkout to the branch the documentation should be build from

3 - Build and deploy the documentation
```
mkdocs gh-deploy
```

To exclude a function named `example_function` implemented in `file_name` from the documentation, add a filter to the configuration options (last two lines) in `api.md`:
```
::: src.geostat.file_name
    options:
        show_root_heading: True
        show_source: True
        filters:
          - "!example_function"
```

To document a class named `class_name` implemented in `file_name` that is not yet documented, add the following lines to `api.md`
```
::: src.geostat.file_name.class_name
    options:
        show_root_heading: True
        show_source: True
```

To document a function named `function_name` implemented in `file_name` that is not yet documented, add the following lines to `api.md`
```
::: src.geostat.file_name.function_name
    options:
        show_root_heading: True
        show_source: True
```
