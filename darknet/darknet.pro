VERSION = 0.1.0
TEMPLATE = app
include($$PWD/../general.pri)

CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

DEFINES += GPU
DEFINES += CUDNN
DEFINES += OPENCV
DEFINES += OPENMP

INCLUDEPATH += \
    $$PWD/../include \
    $$PWD/../src

HEADERS += \
    $$files($$PWD/../src/*.h)

SOURCES += \
    $$files($$PWD/../src/*.c) \
    $$files($$PWD/../examples/*.c)

win32 {
    INCLUDEPATH += $$PWD/../windows
    HEADERS += $$files($$PWD/../windows/*.h)
    SOURCES += $$files($$PWD/../windows/*.c)
}


win32 {
    DEFINES += HAVE_STRUCT_TIMESPEC
    include($$PWD/../ThirdParty/Pthreads.pri)
} else {
    LIBS += -lpthread
}
contains(DEFINES, GPU): include($$PWD/../ThirdParty/CUDA.pri)
contains(DEFINES, GPU): contains(DEFINES, CUDNN): LIBS += -lcudnn
contains(DEFINES, OPENCV): include($$PWD/../ThirdParty/OpenCV.pri)
contains(DEFINES, OPENMP) {
    win32 {
        QMAKE_CFLAGS += /openmp
    } else {
        QMAKE_CFLAGS += -fopenmp
        LIBS += -fopenmp
    }
}
win32: LIBS += -lWs2_32


contains(DEFINES, GPU) {
    CUDA_SOURCES += $$files($$PWD/../src/*.cu)
    NVCC_OPTIONS += \
        -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=[sm_50,compute_50] \
        -gencode arch=compute_52,code=[sm_52,compute_52] \
        $$join(DEFINES, " -D", "-D", "") \
        $$join(INCLUDEPATH, "\" -I\"", "-I\"", "\"") \
        $$LIBS

    CONFIG(debug, debug|release): NVCC_OPTIONS += -D_DEBUG

    win32 {
        CONFIG(debug, debug|release): NVCC_OPTIONS += -Xcompiler /MDd
        else: NVCC_OPTIONS += -Xcompiler /MD
    }

    cuda_compiler.input = CUDA_SOURCES
    cuda_compiler.output = $$DEST_ROOT/darknet/$$CONFIGURATION_NAME/${QMAKE_FILE_BASE}_cuda$$BUILD_OBJECT_FILE_SUFFIX
    cuda_compiler.commands = nvcc $$NVCC_OPTIONS -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_compiler.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_compiler
}


# install
target.path = $$INSTALL_PREFIX/bin
install_data.path = $$INSTALL_PREFIX/bin
install_data.files = $$PWD/../data
install_cfg.path = $$INSTALL_PREFIX/bin
install_cfg.files = $$PWD/../cfg
INSTALLS += target install_data install_cfg
