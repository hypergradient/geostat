## Development and testing

pip install -e .[test]

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
```

2 - Navigate to the GitHub repository and checkout to the main branch

3 - Update the documentation
```
mkdocs gh-deploy
```

