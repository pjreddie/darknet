# OpenCV Lib

OPENCV_ROOT = $$(OPENCV_ROOT)
isEmpty(OPENCV_ROOT) {
    OPENCV_ROOT = $$LIBS_ROOT/OpenCV
}

include($$OPENCV_ROOT/info.pri)  # define "CV_MAJOR_VERSION" and "CV_LIB_POSTFIX"

INCLUDEPATH += $$OPENCV_ROOT/include

win32 {
    CONFIG(debug, debug|release): OPENCV_LIB_POSTFIX = $$CV_LIB_POSTFIX"d"
    else: OPENCV_LIB_POSTFIX = $$CV_LIB_POSTFIX
} else {
}

LIBS += -L$$OPENCV_ROOT/bin
LIBS += -L$$OPENCV_ROOT/lib \
    -lopencv_core$$OPENCV_LIB_POSTFIX \
    -lopencv_highgui$$OPENCV_LIB_POSTFIX \
    -lopencv_imgproc$$OPENCV_LIB_POSTFIX

greaterThan(CV_MAJOR_VERSION, 2) {
    LIBS += \
        -lopencv_imgcodecs$$OPENCV_LIB_POSTFIX \
        -lopencv_videoio$$OPENCV_LIB_POSTFIX
    !win32: CONFIG(debug, debug|release): QMAKE_CFLAGS += -O4 -g
}

# install
install_dependency_opencv.path = $$INSTALL_PREFIX/bin
install_dependency_opencv.files = \
    $$OPENCV_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*opencv_core*$$INSTALL_DEPEND_LIB_FILE_SUFFIX* \
    $$OPENCV_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*opencv_highgui*$$INSTALL_DEPEND_LIB_FILE_SUFFIX* \
    $$OPENCV_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*opencv_imgproc*$$INSTALL_DEPEND_LIB_FILE_SUFFIX* \
    $$OPENCV_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*opencv_imgcodecs*$$INSTALL_DEPEND_LIB_FILE_SUFFIX* \
    $$OPENCV_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*opencv_videoio*$$INSTALL_DEPEND_LIB_FILE_SUFFIX*
INSTALLS += install_dependency_opencv
