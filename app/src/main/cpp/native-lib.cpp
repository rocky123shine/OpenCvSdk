#include <jni.h>

//
// Created by rocky on 2022/7/15.
//
#include "opencv2/opencv.hpp"

using namespace cv;

extern "C"
JNIEXPORT jlong JNICALL
Java_com_rocky_opencv2_Mat_nMat(JNIEnv *env, jobject thiz) {
    Mat *mat = new Mat();
    return reinterpret_cast<jlong>(mat);
}
extern "C"
JNIEXPORT jlong JNICALL
Java_com_rocky_opencv2_Mat_nMatIII(JNIEnv *env, jobject thiz, jint row, jint cols, jint type) {
    Mat *mat = new Mat(row, cols, type);
    return reinterpret_cast<jlong>(mat);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_rocky_opencv2_Mat_nMatAt(JNIEnv *env, jobject thiz, jlong matPtr, jint row, jint cols,
                                  jfloat value) {
    Mat *mat = reinterpret_cast<Mat *>(matPtr);
    mat->at<float>(row, cols) = value;
}
