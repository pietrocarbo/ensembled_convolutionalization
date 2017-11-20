# Progetto di Machine Learning

Download Food dataset from https://www.vision.ee.ethz.ch/datasets_extra/food-101/ unzip and place it in a folder named "dataset-ethz101food".

Download and install Python 3.5 from https://www.python.org/downloads/release/python-350/

Clone this github repository in a folder that will be called "repo_home"


From a terminal run the commands
pip install pipenv
pip install virtualenv
cd repo_home
virtualenv env
./env/Scripts/activate
pip install -r requirements.txt

Move the folder dataset-ethz101food inside "repo_home"

Run the script copy_splitdataset.py to split the dataset in train/test folder (delete the "images" directory left if you want)
