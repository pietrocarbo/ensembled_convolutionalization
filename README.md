# Machine Learning Project

## Setup instructions

Download Food dataset from https://www.vision.ee.ethz.ch/datasets_extra/food-101/ unzip and place it in a folder named `dataset-ethz101food`.

Download and install Python 3.5 from https://www.python.org/downloads/release/python-350/

Clone this github repository in a folder that we will call `repo_home`

From a terminal run the commands
* `pip install pipenv && pip install virtualenv`
* `cd repo_home`
* `virtualenv env` (Mac OS X -> `virtualenv -p python3 env`)
* `./env/Scripts/activate` (or `source env/bin/activate` on Unix systems)
* `pip install -r requirements.txt`

Move the folder `dataset-ethz101food` inside `repo_home`

Run the script `copy_splitdataset.py` to split the dataset in train/test folders (delete the `images` directory left if you want to save disk space)


## TODO
* Trovare il modello pretrained migliore da usare come baseline (o meglio cercare online modello allenato su ethz o altro dataset di cibo)

* Usare i parametri di trasformazione di ImageDataGenerator per ottenere qualcosa in più di accuracy

* Segmentazione tramite Fully Convolutional Networks (modello a parte inizialmente poi da integrare per classificazione foto con bassa confidenza)

###### Comando per aggiornare Keras e Theano
pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps
pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps

###### Flag per usare GPU su Theano backend
conda create -n env1 python python-dev numpy scipy mkl theano pygpu
THEANO_FLAGS=device=cuda0 python <test_theanoGPU.py>
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.7,dnn.enabled=True"