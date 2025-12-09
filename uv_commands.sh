donaldpg@Donalds-MacBook-Pro pyTAAA % 

#uv python install 3.9
#uv  init -v -p 3.9 --cache-dir /tmp --name pyTAAA --app --description "Tactical Asset Allocation Advisor"

cd /Users/donaldpg/PyProjects/PyTAAA.master
uv python install 3.11
uv  init -v -p 3.11 --cache-dir /tmp --name pyTAAA --app --description "Tactical Asset Allocation Advisor"
uv venv --python 3.11
source .venv/bin/activate

uv add scipy pandas matplotlib bs4 tables yfinance paramiko holidays finvizfinance
uv add click
uv add git-filter-repo

uv run pytaaa_main.py --json /Users/donaldpg/PyProjects/PyTAAA.master/pytaaa_hma.json


uv run  PyTAAA.py


uv add spyder
uv run  spyder&






#uv  init -v -p 3.11 --cache-dir /tmp --name pyTAAA --app --description "Tactical Asset Allocation Advisor"

#uv init --script PyTAAA.py --python 3.11
#uv run  PyTAAA.py
#uv add --script PyTAAA.py numpy
#uv add --script PyTAAA.py pandas
#uv add --script PyTAAA.py matplotlib
#uv add --script PyTAAA.py scipy
#uv add --script PyTAAA.py bs4
#uv add --script PyTAAA.py h5py
#uv add --script PyTAAA.py tables

uv sync --script PyTAAA.py


'Donalds-MacBook-Pro.local'


uv venv my_env
source my_env/bin/activate
uv pip install pandas==2.1.0 flask
uv pip list
uv pip install pandas

Another environment with a different version of python
uv venv 3rd_env --python 3.11

deactivate
source my_second_env/bin/activate

see all the environments listed:
user@guest-MacBook-Air my_uv_project % ls 
3rd_env        my_env        my_second_env
If you want to remove one, you can delete the environment folder like this:
rm -rf my_second_env

