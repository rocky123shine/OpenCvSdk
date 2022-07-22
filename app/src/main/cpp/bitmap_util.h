//
// Created by rocky on 2022/7/14.
//

#ifndef OPENCV_BITMAP_UTIL_H
#define OPENCV_BITMAP_UTIL_H
#include <jni.h>
class bitmap_util {
public:
    static jobject create_bitmap(JNIEnv*env,int width,int height,int type);
};


#endif //OPENCV_BITMAP_UTIL_H
