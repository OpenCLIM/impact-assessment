# Flood Impacts DAFNI Model

[![build](https://github.com/OpenCLIM/flood-impacts-dafni/workflows/build/badge.svg)](https://github.com/OpenCLIM/flood-impacts-dafni/actions)

This repo contains the files required to build and test the flood-impacts-dafni model.
All processing steps are contained in [`run.py`](https://github.com/OpenCLIM/Flood-Impact-Model-V2/blob/master/run.py)

## Documentation
[flood-impacts.md](https://github.com/OpenCLIM/Flood-Impact-Model-V2/blob/master/docs/flood-impacts.md)

To build the documentation:
```
cd docs
python build_docs.py
```

## Dependencies
[environment.yml](https://github.com/OpenCLIM/Flood-Impact-Model-V2/blob/master/environment.yml)

## Usage 
`docker build -t flood-impact-model-v2 . && docker run -v "data:/data" --env PYTHONUNBUFFERED=1 --env THRESHOLD=0.1 --name flood-impact-model-v2 flood-impact-model-v2 `

or

`python run.py`
