from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]

TRAIN_DIR = REPO_ROOT/'train'
TEST_DIR = REPO_ROOT/'test'
MODEL_DIR = TRAIN_DIR/'models'

DATASET_ROOT = TRAIN_DIR/'dataset/ESC-50'

CPP_SOURCE_DIR = REPO_ROOT/'inference-cpp'
CPP_BUILD_DIR = CPP_SOURCE_DIR/'build'