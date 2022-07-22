package com.rocky.opencv2

import android.graphics.Bitmap

/**
 * <pre>
 *     author : rocky
 *     time   : 2022/07/15
 *     des    ：java 层的基本功提供
 * </pre>
 */
class Utils {
    companion object {
        @JvmStatic
        fun bitmap2Mat(bitmap: Bitmap, src: Mat) {
            nNitmap2Mat(bitmap, src.mNativePtr)
        }

        private external fun nNitmap2Mat(bitmap: Bitmap, matPtr: Long)
        fun mat2Bitmap(dst: Mat, bitmap: Bitmap) {
            nMat2Bitmap(dst.mNativePtr, bitmap)
        }

        private external fun nMat2Bitmap(matPtr: Long, bitmap: Bitmap)

    }

}