import os

cwd = os.getcwd()

if not os.path.exists('model'):
    os.mkdir('model')

if not os.path.exists(f'{cwd}/result'):
    os.mkdir(f'{cwd}/result')

TRAIN_DATASET_PATH = f'{cwd}/dataset/train'
VALID_DATASET_PATH = f'{cwd}/dataset/valid'
TEST_DATASET_PATH = f'{cwd}/dataset/test'
MODEL_PATH = f'{cwd}/model'

MODEL = 'efficientdet_lite0'
MODEL_NAME = 'hardware_supplies.tflite'
CLASSES = ['Bistay sand', 'Cement', 'Gravel', 'Hollow blocks', 'Rebar', 'Sack', 'Skim coat']
EPOCHS = 5
BATCH_SIZE = 4