#!/bin/bash

# Setup script for Cloud Environment (Colab/Cloud with T4 GPU)

# 1. Update package list
sudo apt-get update

# 2. Install graphviz and its dependencies (for pygraphviz if needed)
sudo apt-get install -y graphviz libgraphviz-dev pkg-config

# 3. Clone pgmpy dev branch
if [ ! -d "pgmpy" ]; then
    git clone -b dev https://github.com/pgmpy/pgmpy.git
fi

# 4. Install pgmpy in editable mode
cd pgmpy
pip install -e ".[all]"
cd ..

# 5. Install other requirements
pip install -r requirements.txt

# 6. Verify installation
python -c "import pgmpy; print(f'pgmpy {pgmpy.__version__} is ready on Colab!')"
