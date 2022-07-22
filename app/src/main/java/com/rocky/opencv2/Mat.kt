package com.rocky.opencv2

/**
 * <pre>
 *     author : rocky
 *     time   : 2022/07/15
 * </pre>
 */
class Mat(var row: Int = 0, var cols: Int = 0, var type: CVType?) {
    var mNativePtr: Long = 0//对应的native层Mat地址

    constructor() : this(0, 0, null)

    init {
        //需要在native 申请一块内存
        mNativePtr = if (row == 0 || cols == 0 || type == null) {
            nMat()
        } else {
            nMatIII(row, cols, type!!.value)
        }

    }

     private external fun nMat(): Long
    private external fun nMatIII(row: Int, cols: Int, type: Int): Long
    fun at(row: Int, cols: Int, value: Float) {
        nMatAt(mNativePtr, row, cols, value)
    }

    private external fun nMatAt(mNativePtr: Long, row: Int, cols: Int, value: Float)

}