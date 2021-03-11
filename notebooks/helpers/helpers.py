from .dirs import *

def write_vocab_cpp_header (vocab):
    """Write C++ header file to containing a std::vector with the elements from `vocab`"""
    with open(CPP_SOURCE_DIR/'src/vocab-gen.h', 'w') as f:
        f.write('// This is a generated file. Manual changes will be overwritten!!!\n')
        f.write('#include <vector>\n#include <string>\n#pragma once\n\nconst std::vector<std::string> vocab {{\n')
        f.writelines([f'\t{{\"{word}\"}},\n' for word in vocab])
        f.write('}};')

