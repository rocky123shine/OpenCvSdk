package com.rocky.opencvsdk

import android.graphics.Bitmap
import com.rocky.opencv2.CVType
import com.rocky.opencv2.Imgproc
import com.rocky.opencv2.Mat
import com.rocky.opencv2.Utils

/**
 * <pre>
 *     author : rocky
 *     time   : 2022/07/15
 * </pre>
 */
class OpencvUtils {
    companion object {

        @JvmStatic
        fun mask(bitmap: Bitmap): Bitmap {
            val src = Mat()
            val dst = Mat()
            Utils.bitmap2Mat(bitmap, src)
            val kernel = Mat(3, 3, CVType.CV_32FC1)

//            //使用掩膜 给kernel赋值
            kernel.at(0, 0, 0f)
            kernel.at(0, 1, -1f)
            kernel.at(0, 2, 0f)

            kernel.at(1, 0, -1f)
            kernel.at(1, 1, 5f)
            kernel.at(1, 2, -1f)

            kernel.at(2, 0, 0f)
            kernel.at(2, 1, -1f)
            kernel.at(2, 2, 0f)


            Imgproc.filter2D(src, dst, kernel)
           val bitmap = createBitmap(dst.mNativePtr);
            Utils.mat2Bitmap(dst, bitmap)
            return bitmap
        }

        private external fun createBitmap(dstPtr:Long): Bitmap
    }
}