cmake_minimum_required(VERSION 3.18.1)
project("mOpencv")

#配置opencv
#配置加载native依赖->引入OpenCV头文件
# 添加头文件 正常配置一个./include/opencv2即可 但是 ./include/opencv2文件 需要其他文件 所有多引入前两个
include_directories(./include ./include/opencv ./include/opencv2)


#动态方式加载
#引入libopencv_java3.so文件
add_library(opencv_java3
        SHARED
        IMPORTED)

message(====== ${CMAKE_PROJECT_HOMEPAGE_URL})

# 无法使用相对路径  暂无解决办法
set_target_properties(opencv_java3
        PROPERTIES IMPORTED_LOCATION
        /Users/jake/Desktop/android/code/study/OpenCvSdk/app/libs/${ANDROID_ABI}/libopencv_java3.so)


add_library(
        mopencv

        SHARED
        native-lib.cpp
        cv_helper.cpp
        bitmap_util.cpp
        native_utils.cpp
        NativeImgproc.cpp)

find_library(
        log-lib
        log)

target_link_libraries(
        mopencv
        opencv_java3
        jnigraphics
        ${log-lib})