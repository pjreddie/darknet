# Pthreads Lib

PTHREADS_ROOT = $$(PTHREADS_ROOT)
isEmpty(PTHREADS_ROOT) {
    PTHREADS_ROOT = $$LIBS_ROOT/Pthreads
}

INCLUDEPATH += $$PTHREADS_ROOT/include

LIBS += -L$$PTHREADS_ROOT/bin
LIBS += -L$$PTHREADS_ROOT/lib \
    -lpthreadVC2

# install
install_dependency_pthreads.path = $$INSTALL_PREFIX/bin
install_dependency_pthreads.files = $$PTHREADS_ROOT/$$INSTALL_DEPEND_LIB_DIR_NAME/*pthread*$$INSTALL_DEPEND_LIB_FILE_SUFFIX*
INSTALLS += install_dependency_pthreads
