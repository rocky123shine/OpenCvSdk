//
// Created by rocky on 2022/7/15.
//

#include <jni.h>
#include "cv_helper.h"
#include <jni.h>
#include <jni.h>
#include <jni.h>

using namespace cv;

extern "C"
JNIEXPORT void JNICALL
Java_com_rocky_opencv2_Utils_00024Companion_nNitmap2Mat(JNIEnv *env, jobject thiz, jobject bitmap,
                                                        jlong matPtr) {
    Mat *mat = reinterpret_cast<Mat *>(matPtr);
    cv_helper::bitmap2mat(env, bitmap, *mat);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_rocky_opencv2_Utils_00024Companion_nMat2Bitmap(JNIEnv *env, jobject thiz, jlong matPtr,
                                                        jobject bitmap) {
    Mat *mat = reinterpret_cast<Mat *>(matPtr);
    cv_helper::mat2bitmap(env, *mat, bitmap);

}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_rocky_opencvsdk_OpencvUtils_00024Companion_createBitmap(JNIEnv *env, jobject thiz,
                                                                 jlong dstPtr) {
    Mat *mat = reinterpret_cast<Mat *>(dstPtr);
   return cv_helper::createBitmap(env, mat->cols, mat->rows, 0);
}