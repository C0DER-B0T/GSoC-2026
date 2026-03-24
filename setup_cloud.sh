#!/bin/bash

# Setup script for Cloud Environment (Colab/Cloud with T4 GPU)

# 1. Update package list
sudo apt-get update

# 2. Install graphviz and its dependencies (for pygraphviz if needed)
sudo apt-get install -y graphviz libgraphviz-dev pkg-config

# 3. Clone pgmpy dev branch into a uniquely named folder to avoid import collisions
if [ ! -d "pgmpy_source" ]; then
    git clone -b dev https://github.com/pgmpy/pgmpy.git pgmpy_source
fi

# 4. Install pgmpy in editable mode from the source folder
cd pgmpy_source
pip install -e ".[all]"
cd ..

# 5. Install other requirements
pip install -r requirements.txt

# 6. Verify installation
python -c "import pgmpy; print(f'pgmpy {pgmpy.__version__} is ready on Colab!')"
