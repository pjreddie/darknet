isEmpty(SOLUTION_DIR) {
    SOLUTION_DIR = $$PWD
#    warning("Variable \"SOLUTION_DIR\" is not defined, so the default value \"$$SOLUTION_DIR\" (the directory of \"general.pri\") will be used.")
}


isEmpty(SOLUTION_NAME) {
    SOLUTION_NAME = $$basename(PWD)
#    warning("Variable \"SOLUTION_NAME\" is not defined, so the default value \"$$SOLUTION_NAME\" (the directory name of \"general.pri\") will be used.")
}


isEmpty(PROJECT_DIR) {
    PROJECT_DIR = $$_PRO_FILE_PWD_
}


isEmpty(PROJECT_NAME) {
    PROJECT_NAME = $$basename(_PRO_FILE_PWD_)
}


isEmpty(VERSION) {
    VERSION = 0.0.0
    warning("Variable \"VERSION\" is not defined, so the default value \"0.0.0\" will be used.")
}


isEmpty(CONFIGURATION_NAME) {
    CONFIG(debug, debug|release): CONFIGURATION_NAME = debug
    else: CONFIGURATION_NAME = release
}


isEmpty(COMPILER_NAME) {
    COMPILER_NAME = $$(COMPILER_NAME)
    isEmpty(COMPILER_NAME) {
        win32: COMPILER_NAME = msvc
        else: COMPILER_NAME = gcc
    }
}


isEmpty(ARCHITECTURE_NAME) {
    ARCHITECTURE_NAME = $$(ARCHITECTURE_NAME)
    isEmpty(ARCHITECTURE_NAME) {
        ARCHITECTURE_NAME = x64
    }
}


isEmpty(TEMP_DIR) {
    TEMP_DIR = $$(TEMP_DIR)
    isEmpty(TEMP_DIR) {
        win32: TEMP_DIR = E:/temp
        else: TEMP_DIR = $$(HOME)/temp
    }
}


isEmpty(DEST_ROOT) {
    DEST_ROOT = $$(DEST_ROOT)
    isEmpty(DEST_ROOT) {
        DEST_ROOT = $$TEMP_DIR/$$SOLUTION_NAME/build/$$SOLUTION_NAME-$$VERSION-$$COMPILER_NAME-$$ARCHITECTURE_NAME
    }
}


isEmpty(INSTALL_PREFIX) {
    INSTALL_PREFIX = $$(INSTALL_PREFIX)
    isEmpty(INSTALL_PREFIX) {
        INSTALL_PREFIX = $$TEMP_DIR/$$SOLUTION_NAME/install/$$SOLUTION_NAME-$$VERSION-$$COMPILER_NAME-$$ARCHITECTURE_NAME
    }
}


isEmpty(LIBS_ROOT) {
    LIBS_ROOT = $$(LIBS_ROOT)
    isEmpty(LIBS_ROOT) {
        LIBS_ROOT = $$clean_path($$SOLUTION_DIR/../$$SOLUTION_NAME"_"ThirdParty)
    }
}


isEmpty(OTHERFILES_ROOT) {
    OTHERFILES_ROOT = $$(OTHERFILES_ROOT)
    isEmpty(OTHERFILES_ROOT) {
        OTHERFILES_ROOT = $$clean_path($$SOLUTION_DIR/../$$SOLUTION_NAME"_"OtherFiles)
    }
}


win32 {
    BUILD_DEPEND_LIB_DIR_NAME = lib
    BUILD_DEPEND_LIB_FILE_SUFFIX_SHARED = "."lib
    BUILD_DEPEND_LIB_FILE_SUFFIX_STATIC = "."lib
    BUILD_DEPEND_LIB_FILE_SUFFIX = $$BUILD_DEPEND_LIB_FILE_SUFFIX_SHARED
    INSTALL_DEPEND_LIB_DIR_NAME = bin
    BUILD_OBJECT_FILE_SUFFIX = "."obj
    INSTALL_DEPEND_LIB_FILE_SUFFIX = "."dll

    deployqt = windeployqt

    EXECUTABLE_FILE_SUFFIX = "."exe
    EXECUTABLE_SCRIPT_FILE_SUFFIX = "."bat
} else {
    BUILD_DEPEND_LIB_DIR_NAME = lib
    BUILD_DEPEND_LIB_FILE_SUFFIX_SHARED = "."so
    BUILD_DEPEND_LIB_FILE_SUFFIX_STATIC = "."a
    BUILD_DEPEND_LIB_FILE_SUFFIX = $$BUILD_DEPEND_LIB_FILE_SUFFIX_SHARED
    BUILD_OBJECT_FILE_SUFFIX = "."o
    INSTALL_DEPEND_LIB_DIR_NAME = lib
    INSTALL_DEPEND_LIB_FILE_SUFFIX = "."so

    deployqt = linuxdeployqt

    EXECUTABLE_FILE_SUFFIX =
    EXECUTABLE_SCRIPT_FILE_SUFFIX = "."sh
}


defineReplace(getInstallDeployQtCommand) {
    TARGET_ = $$1
    win32 {
        return($$deployqt --dir $$INSTALL_PREFIX --libdir $$INSTALL_PREFIX/lib/Qt --plugindir $$INSTALL_PREFIX/plugins $$INSTALL_PREFIX/bin/$$TARGET_$$EXECUTABLE_FILE_SUFFIX)
    } else {
        LIBS_ = $$2
        DEPEND_LIB_PATHS =
        for (lib_dir, LIBS_) {
            lib_dir = $$replace(lib_dir, -L, "")
            exists($$lib_dir/*) {
                DEPEND_LIB_PATHS += $$lib_dir
            }
        }
        setup_env = $$join(DEPEND_LIB_PATHS, ":", "export LD_LIBRARY_PATH=\"", ":$LD_LIBRARY_PATH\"")

        return($$setup_env; $$deployqt $$INSTALL_PREFIX/bin/$$TARGET_$$EXECUTABLE_FILE_SUFFIX)
    }
}


isEmpty(DEFINED_TARGET) {
    CONFIG(debug, debug|release): TARGET = $$PROJECT_NAME"d"
    else: TARGET = $$PROJECT_NAME

    DEFINED_TARGET = 1
}
isEmpty(DEFINED_DESTDIR) {
    equals(TEMPLATE, "lib") {
        DESTDIR = $$DEST_ROOT/$$PROJECT_NAME/lib
    } else: equals(TEMPLATE, "app") {
        DESTDIR = $$DEST_ROOT/$$PROJECT_NAME/bin
    }

    DEFINED_DESTDIR = 1
}
