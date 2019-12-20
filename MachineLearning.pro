TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        src\main.cpp

HEADERS += \
    src/decisiontree/decisiontree.h \
    src/demo/decisiontreedemo.h \
    src/demo/orl.h \
    src/demo/watermelon.h \
    src/regression/linearregression.h \
    src/regression/logitregression.h \
    src/stdafx.h

PRECOMPILED_HEADER += src/stdafx.h

INCLUDEPATH += D:\Programs\Eigen\headers

INCLUDEPATH += D:\Programs\OpenCV\opencv-4.1.2-built-Qt\install\include

LIBS += D:\Programs\OpenCV\opencv-4.1.2-built-Qt\bin\libopencv_core412.dll
LIBS += D:\Programs\OpenCV\opencv-4.1.2-built-Qt\bin\libopencv_highgui412.dll
LIBS += D:\Programs\OpenCV\opencv-4.1.2-built-Qt\bin\libopencv_imgcodecs412.dll
LIBS += D:\Programs\OpenCV\opencv-4.1.2-built-Qt\bin\libopencv_imgproc412.dll
LIBS += D:\Programs\OpenCV\opencv-4.1.2-built-Qt\bin\libopencv_features2d412.dll
LIBS += D:\Programs\OpenCV\opencv-4.1.2-built-Qt\bin\libopencv_calib3d412.dll
