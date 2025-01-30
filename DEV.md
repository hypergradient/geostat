## Development and testing

pip install -e .[test]

## Deploy to PyPI:

To build:
```
python -m build
```
To upload to PyPi:
```
twine upload --repository geostat
```
