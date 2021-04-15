# LoadShedderInterface

## Requirements
- Use Ubuntu 18.04 (other linux distros could work, but the requirements scripts may not work)
- All the requirements can be install using the ansible script located in ./requirements.
  - See README in requirements directory

## Code Structure
- The src directory contains all the source code.
- The src/capnp\_serial contains the schema definition for the features and other objects that need to be serialized.

## Example code
- The file ./example\_python3.py  contains an example of how to serialize and deserialize files using the given schema in src/capnp\_serial.
- To run it:
```bash
python3 ./example_python.py
```

