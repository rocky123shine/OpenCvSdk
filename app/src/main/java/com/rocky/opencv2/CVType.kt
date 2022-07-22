package com.rocky.opencv2

/**
 * Created by hcDarren on 2019/3/2.
 * Type 类型 ，value 对应 Mat.cpp 的 type 类型
 */
enum class CVType(val value: Int) {
    CV_8UC1(0), CV_8UC2(8), CV_8UC4(24), CV_32FC1(5);
}