# Building the C++ application

The C++ application can be built directly from within the notebook tutorials.
Of course, you can also build it outside of the notebooks with

```
mkdir inference-cpp/build && cd inference-cpp/build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch; print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release
```

Note that the C++ application does not have any dependency on Python. It does
depend, however, on LibTorch, the C++ implementation of PyTorch. PyTorch
itself uses LibTorch under the hood, so by installing PyTorch we already have
all the header, library and cmake configuration files we need to build our
C++ application. We can just use the `torch.utils.cmake_prefix_path` variable
to make sure cmake finds everything.

PyTorch also provides distributions of LibTorch on its own. These can be used
in environments where the whole Python side is not needed (e.g. building
applications for deployment)