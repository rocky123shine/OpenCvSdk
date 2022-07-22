package com.rocky.opencv2

/**
 * <pre>
 *     author : rocky
 *     time   : 2022/07/15
 * </pre>
 */
class Imgproc {
    companion object {
        @JvmStatic
        fun filter2D(
            src: Mat,
            dst: Mat,
            kernel: Mat,
        ) {
            nFilter2D(src.mNativePtr, dst.mNativePtr, kernel.mNativePtr)
        }

        private external fun nFilter2D(mNativePtr: Long, mNativePtr1: Long, mNativePtr2: Long)
    }
}