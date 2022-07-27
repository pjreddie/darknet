# CUDA Lib

CUDA_ROOT = $$(CUDA_ROOT)
isEmpty(CUDA_ROOT) {
    CUDA_ROOT = $$LIBS_ROOT/CUDA
}

INCLUDEPATH += $$CUDA_ROOT/include

LIBS += -L$$CUDA_ROOT/bin
LIBS += -L$$CUDA_ROOT/nvvm/bin
LIBS += -L$$CUDA_ROOT/lib/x64 \
    -lcuda \
    -lcudart \
    -lcublas \
    -lcurand

# install
install_dependency_cuda.path = $$INSTALL_PREFIX/bin
install_dependency_cuda.files = \
    $$CUDA_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*cudart*$$INSTALL_DEPEND_LIB_FILE_SUFFIX* \
    $$CUDA_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*cublas*$$INSTALL_DEPEND_LIB_FILE_SUFFIX* \
    $$CUDA_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*curand*$$INSTALL_DEPEND_LIB_FILE_SUFFIX* \
    $$CUDA_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*cudnn*$$INSTALL_DEPEND_LIB_FILE_SUFFIX*
INSTALLS += install_dependency_cuda
