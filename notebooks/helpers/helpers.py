
from .dirs import *

import json

def load_params(params_path:Path=None):
    """Load parameter dictionary from JSON file"""
    json_path = params_path if params_path is not None else TRAIN_DIR/'params.json'
    assert json_path.is_file(), f'No json configuration file found at {json_path}'
    with open(json_path, 'r') as infile:
        json_data = json.load(infile)
    return json_data

def save_params(data, params_path:Path=None):
    """Save parameter dictionary to JSON file"""
    json_path = params_path if params_path is not None else TRAIN_DIR/'params.json'
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)

def write_vocab_cpp_header (vocab):
    """Write C++ header file to containing a std::vector with the elements from `vocab`"""
    with open(CPP_SOURCE_DIR/'src/vocab-gen.h', 'w') as f:
        f.write('// This is a generated file. Manual changes will be overwritten!!!\n')
        f.write('#include <vector>\n#include <string>\n#pragma once\n\nconst std::vector<std::string> vocab {{\n')
        f.writelines([f'\t{{\"{word}\"}},\n' for word in vocab])
        f.write('}};')

