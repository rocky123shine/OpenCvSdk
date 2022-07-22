#include <jni.h>
#include "opencv.hpp"

using namespace cv;
using namespace std;

#include "android/log.h"

void HSV(const Mat *src, Mat *dst) {//拆分
    vector<Mat> bgr_s;
    split(*src, bgr_s);
    /**
     * const Mat* images,-----------》输入图片
     * int nimages, ----------------》输入图像的个数
     * const int* channels,---------》相应图像的第几通道
     * InputArray mask,-------------》掩膜
     * OutputArray hist,------------》输出Mat
     * int dims, -------------------》需要统计的通道个数
     * const int* histSize,---------》等级个数  即直方图分堆个数
     * const float** ranges,--------》 数据范围 brg - 0到255 hsv 0到360 0到255 0 到255 注意和格式匹配
     * bool uniform = true,---------》
     * bool accumulate = false------》
     */
    int histSize = 256;
    float range[] = {0, 255};
    const float *ranges = {range};
    Mat hist_b, hist_g, hist_r;

    calcHist(&bgr_s[0], 1, 0, Mat(), hist_b, 1, &histSize, &ranges, true, false);
    calcHist(&bgr_s[1], 1, 0, Mat(), hist_g, 1, &histSize, &ranges, true, false);
    calcHist(&bgr_s[2], 1, 0, Mat(), hist_r, 1, &histSize, &ranges, true, false);
    //归一化
/**
 * InputArray src,
 * InputOutputArray dst,
 * double alpha = 1, -------------------》最小值
 * double beta = 0,---------------------》最大值
 * int norm_type = NORM_L2,
 * int dtype = -1,
 * InputArray mask = noArray()
 */
    int hist_h = 400;
    int hist_w = 1024;
    int bin_w = hist_w / histSize;
    normalize(hist_b, hist_b, 0, hist_h, NORM_MINMAX, -1, Mat());
    normalize(hist_g, hist_g, 0, hist_h, NORM_MINMAX, -1, Mat());
    normalize(hist_r, hist_r, 0, hist_h, NORM_MINMAX, -1, Mat());

    //画到Mat上
    Mat histImag(hist_h, hist_w, src->type(), Scalar(0, 0, 0));


    for (int i = 1; i < histSize; ++i) {
        line(
                histImag,
                Point((i - 1) * bin_w, (int) (hist_h * 1.0f - hist_b.at<float>(i - 1))),
                Point(i * bin_w, (int) (hist_w - hist_b.at<float>(i))),
                Scalar(255, 0, 0),
                bin_w,
                LINE_AA);
        line(
                histImag,
                Point((i - 1) * bin_w, (int) (hist_h * 1.0f - hist_g.at<float>(i - 1))),
                Point(i * bin_w, (int) (hist_w - hist_g.at<float>(i))),
                Scalar(0, 255, 0),
                bin_w,
                LINE_AA);
        line(
                histImag,
                Point((i - 1) * bin_w, (int) (hist_h * 1.0f - hist_r.at<float>(i - 1))),
                Point(i * bin_w, (int) (hist_w - hist_r.at<float>(i))),
                Scalar(0, 0, 255),
                bin_w,
                LINE_AA);
    }
    cvtColor(histImag, *dst, COLOR_BGRA2BGR);
}


void filter2d(const Mat *src, Mat *dst, const Mat *kernel) {
    Mat bgr;
    cvtColor(*src, bgr, CV_BGRA2GRAY);
    filter2D(bgr, *dst, src->depth(), *kernel);
}

void crcb(const Mat *src, Mat *dst) {
    Mat ycrcb;
    cvtColor(*src, ycrcb, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    //拆分颜色通道
    split(ycrcb, channels);
    //直方均衡
    equalizeHist(channels[0], channels[0]);
//    equalizeHist(channels[1], channels[1]);
//    equalizeHist(channels[2], channels[2]);
//合并
    merge(channels, ycrcb);
    //在转化回来
    cvtColor(ycrcb, *dst, COLOR_YCrCb2BGR);
}

void equalize(const Mat *src, Mat *dst) {//使用直方均衡
    Mat gray;
    cvtColor(*src, gray, CV_BGRA2GRAY);
//灰度图的直方均衡
    equalizeHist(gray, *dst);
}

void blur(const Mat *src, Mat *dst) {
    Mat blur;
    GaussianBlur(*src, blur, Size(5, 5), BORDER_DEFAULT, BORDER_DEFAULT);
    // 边缘梯度增强（保存图片）x,y 增强
    Mat gard_x, gard_y;
    Scharr(blur, gard_x, CV_32F, 1, 0);
    Scharr(blur, gard_y, CV_32F, 0, 1);
    Mat abs_gard_x, abs_gard_y;
    convertScaleAbs(gard_x, abs_gard_x);
    convertScaleAbs(gard_y, abs_gard_y);
    Mat gard;
    addWeighted(abs_gard_x, 0.5, abs_gard_y, 0.5, 0, gard);

    //二值化
    Mat gray;
    cvtColor(gard, gray, COLOR_BGRA2GRAY);
    Mat binary;
    threshold(gray, *dst, 100, 255, THRESH_BINARY);
}

void backProject(const Mat *src, Mat *dst) {//使用HSV堆H S操作
    Mat hsv;
    cvtColor(*src, hsv, COLOR_BGR2HSV);
    //拆分
    vector<Mat> hsv_s;
    split(hsv, hsv_s);
    float range[] = {0, 180};
    const float *ranges[] = {range};
    int histSize = 2;
    Mat hist;
    calcHist(&hsv_s[0], 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
    //归一化
    normalize(hist, hist, 0, 255, NORM_MINMAX);
    /**
        * const Mat* images,
        * int nimages,
        * const int* channels,
        * InputArray hist,
        * OutputArray backProject,
        * const float** ranges,
        * double scale = 1,
        * bool uniform = true
        */
    calcBackProject(&hsv_s[0], 1, 0, hist, *dst, ranges);
}

void hist_2(const Mat *src, Mat *dst) {
    Mat hsv;
    cvtColor(*src, hsv, CV_BGR2HSV);
    vector<Mat> hist_s;
    split(hsv, hist_s);
    //只改变亮度
    equalizeHist(hist_s[2], hist_s[2]);
    merge(hist_s, hsv);
    cvtColor(hsv, *dst, CV_HSV2BGR);
}


void findContoursss(const Mat *src, Mat *dst) {
    Mat grey;
    cvtColor(*src, grey, CV_BGRA2GRAY);
    //梯度二值化
    Canny(grey, *dst, 50, 150);
    vector<vector<Point>> contours;
    findContours(*dst, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    Mat contours_mat = Mat::zeros(src->size(), CV_8UC3);
    for (int i = 0; i < contours.size(); ++i) {
        drawContours(contours_mat, contours, i, Scalar(255, 0, 255), 1);

    }
    *dst = contours_mat;
}

void hsv_v(Mat *src, Mat *dst) {//线性处理方式
    *dst = src->clone();
    int rows = src->rows;
    int cols = src->cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Vec4b pixels = src->at<Vec4b>(i, j);
            dst->at<Vec4b>(i, j)[0] = saturate_cast<uchar>(pixels[0] + 20);
            dst->at<Vec4b>(i, j)[1] = saturate_cast<uchar>(pixels[1] + 20);
            dst->at<Vec4b>(i, j)[2] = saturate_cast<uchar>(pixels[2] + 20);
            dst->at<Vec4b>(i, j)[3] = 255;

        }
    }
}

void calcHist(const Mat &src, Mat &dest) {
    //存成单通道 int类型CV_32S      8UC4 是多通道  按int 类型读取
    dest.create(1, 256, CV_32S);
    //初始化为黑色
    for (int i = 0; i < 256; ++i) {
        dest.at<int>(i) = 0;
    }
    int rows = src.rows;
    int cols = src.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = src.at<uchar>(i, j);//值一定在256以内
            //统计像素值出现的次数
            dest.at<int>(0, index) += 1;
        }
    }
}

// 手写直方图计算源码
void calcHist_111(const Mat &mat, Mat &hist) {
    // int 存
    hist.create(1, 256, CV_32S);
    for (int i = 0; i < hist.cols; i++) {
        hist.at<int>(0, i) = 0;
    }

    for (int row = 0; row < mat.rows; row++) {
        for (int col = 0; col < mat.cols; col++) {
            // 灰度等级的角标
            int index = mat.at<uchar>(row, col);
            hist.at<int>(0, index) += 1;
        }
    }
}

void normalize(const Mat &src, Mat &dest, int max) {
    //把值缩放到0到最大值之间
    int max_value = 0;
    int rows = src.rows;
    int cols = src.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            //这里 使用int 正常 通用方法 需要判断类型
            int value = src.at<int>(i, j);
            max_value = cv::max(max_value, max);


        }
    }
    dest.create(src.size(), src.type());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            //这里 使用int 正常 通用方法 需要判断类型
            int value = src.at<int>(i, j);
            dest.at<int>(i, j) = (1.0 / max_value) * value * max;
        }
    }

}

void mEqualizeHist(const Mat &src, Mat &dst) {
    //灰度转化
    // 1. 直方图统计
    Mat hist;
    calcHist(src, hist);//手写统计  记录像素出现的次数
    /*
        //cout << hist << endl;
        //绘制直方图
        //归一化 自己写
        //normalize(hist,hist,0,255,NORM_MINMAX);
        normalize(hist, hist, 255);

        //cout << "-->" << hist << endl;
        //绘制直方图 显示
        int bins_w = 10;
        Mat hist_mat(256, bins_w * 256, CV_8UC3);//显示直方图
        for (int i = 0; i < 256; i+=2) {
            Point start(i * bins_w, hist_mat.rows);
            Point end(i * bins_w, hist_mat.rows - hist.at<int>(0, i));
            line(hist_mat, start, end, Scalar(100, 100, 100), bins_w, LINE_AA);
        }
        *dst = hist_mat;
    */
    // 2. 计算直方图中像素的概率
    Mat prob_mat(hist.size(), CV_32FC1);
    // 图片的像素点大小
    float image_size = src.cols * src.rows;
    for (int i = 0; i < hist.cols; i++) {
        float times = hist.at<int>(0, i);
        float prob = times / image_size;
        prob_mat.at<float>(0, i) = prob;
    }
    // 计算累加概率 256
    float prob_sum = 0;
    for (int i = 0; i < prob_mat.cols; i++) {
        float prob = prob_mat.at<float>(0, i);
        prob_sum += prob;
        prob_mat.at<float>(0, i) = prob_sum;
    }

    // 3. 生成一张映射表
    // 生成映射表
    Mat map(hist.size(), CV_32FC1);
    for (int i = 0; i < prob_mat.cols; i++) {
        float prob = prob_mat.at<float>(0, i);
        map.at<float>(0, i) = prob * 255;
    }
    // 4. 从映射表中查找赋值
    int rows = src.rows;
    int cols = src.cols;
    dst.create(src.size(), src.type());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar pixels = src.at<uchar>(i, j);
            dst.at<uchar>(i, j) = map.at<float>(0, pixels);
        }
    }
}

void equalizeHist_111(const Mat &src, Mat &dst) {

    Mat hist;
    calcHist_111(src, hist);
    // 2. 计算直方图中像素的概率
    Mat prob_mat(hist.size(), CV_32FC1);
    // 图片的像素点大小
    float image_size = src.cols * src.rows;
    for (int i = 0; i < hist.cols; i++) {
        float times = hist.at<int>(0, i);
        float prob = times / image_size;
        prob_mat.at<float>(0, i) = prob;
    }
    // 计算累加概率 256 （可读）
    float prob_sum = 0;
    for (int i = 0; i < prob_mat.cols; i++) {
        float prob = prob_mat.at<float>(0, i);
        prob_sum += prob;
        prob_mat.at<float>(0, i) = prob_sum;
    }
    // 生成映射表
    Mat map(hist.size(), CV_32FC1);
    for (int i = 0; i < prob_mat.cols; i++) {
        float prob = prob_mat.at<float>(0, i);
        map.at<float>(0, i) = prob * 255;
    }

    dst.create(src.size(), src.type());

    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            uchar pixels = src.at<uchar>(row, col);
            uchar pixel = map.at<float>(0, pixels);
            dst.at<uchar>(row, col) = pixel;
        }
    }

}

int getBlockSum(const Mat &sum_mat, int col0, int row0, int col1, int row1, int ch) {
    //使用积分和 求和
    int lt = sum_mat.at<Vec3i>(row0, col0)[ch];
    int lb = sum_mat.at<Vec3i>(row1, col0)[ch];
    int rt = sum_mat.at<Vec3i>(row0, col1)[ch];
    int rb = sum_mat.at<Vec3i>(row1, col1)[ch];
    return rb - rt - lb + lt;
}

float getBlockSqSum(const Mat &sq_mat, int col0, int row0, int col1, int row1, int ch) {
    //使用积分和 求和
    float lt = sq_mat.at<Vec3f>(row0, col0)[ch];
    float lb = sq_mat.at<Vec3f>(row1, col0)[ch];
    float rt = sq_mat.at<Vec3f>(row0, col1)[ch];
    float rb = sq_mat.at<Vec3f>(row1, col1)[ch];
    return rb - rt - lb + lt;
}

void fatsBilateralBlur(const Mat &src, Mat &dst, int size) {
    //这里使用了积分图算法 累加和 有助于算法的性能的提升
    //要求出积分图 需要在原mat 的基础上 加上模糊半径行和模糊半径列
    Mat mat;
    int radius = size >> 1;
    //追加边缘行列
    copyMakeBorder(src, mat, radius, radius, radius, radius, BORDER_DEFAULT);
    //求出积分图
    Mat sum_mat, sq_mat;//和的积分图  和平方和的积分图
    integral(mat, sum_mat, sq_mat, CV_32S, CV_32F);

    //给目标赋值
    dst.create(src.size(), src.type());
    int img_w = src.cols;
    int img_h = src.rows;
    int areas = size * size;//这是 kernel的像素数 卷积核的像素数
    int channels = src.channels();

    // 求四个点，左上，左下，右上，右下
    int col0 = 0, row0 = 0, col1 = 0, row1 = 0;
    for (int row = 0; row < img_h; ++row) {
        row0 = row;
        row1 = row0 + size;
        for (int col = 0; col < img_w; ++col) {
            //在这里给dst 赋值
            col0 = col;
            col1 = col0 + size;
            //按照通道 一一赋值

            for (int i = 0; i < channels; ++i) {
                int sum = getBlockSum(sum_mat, col0, row0, col1, row1, i);
                float sq_sum = getBlockSqSum(sq_mat, col0, row0, col1, row1, i);
                //求出方差
                float var_sq = (sq_sum - (sum * sum) * 1.0f / areas) * 1.0f / areas;
                //求出K
                int sigma = size * size;
                float K = var_sq / (var_sq + sigma);
                //带入公式  局部均方差滤波公式  dstPix = （1-k）*m + k*srxPix
                int pixels = src.at<Vec3b>(row, col)[i];
                pixels = (1 - K) * (sum * 1.0 / areas) + K * pixels;
                dst.at<Vec3b>(row, col)[i] = saturate_cast<uchar>(pixels);
            }


        }

    }

}
void skinDetect(const Mat &src, Mat &skinMask){
    skinMask.create(src.size(), CV_8UC1);
    int rows = src.rows;
    int cols = src.cols;

    Mat ycrcb;
    cvtColor(src, ycrcb, COLOR_BGR2YCrCb);

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            Vec3b pixels = ycrcb.at<Vec3b>(row, col);
            uchar y = pixels[0];
            uchar cr = pixels[1];
            uchar cb = pixels[2];

            if (y>80 && 85<cb<135 && 135<cr<180){
                skinMask.at<uchar>(row, col) = 255;
            }
            else{
                skinMask.at<uchar>(row, col) = 0;
            }
        }
    }
}
void fuseSkin(const Mat &src, const  Mat &blur_mat, Mat &dst, const Mat &mask){
    // 融合？
    dst.create(src.size(),src.type());
    GaussianBlur(mask, mask, Size(3, 3), 0.0);
    Mat mask_f;
    mask.convertTo(mask_f, CV_32F);
    normalize(mask_f, mask_f, 1.0, 0.0, NORM_MINMAX);

    int rows = src.rows;
    int cols = src.cols;
    int ch = src.channels();

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            // mask_f (1-k)
            /*
            uchar mask_pixels = mask.at<uchar>(row,col);
            // 人脸位置
            if (mask_pixels == 255){
                dst.at<Vec3b>(row, col) = blur_mat.at<Vec3b>(row, col);
            }
            else{
                dst.at<Vec3b>(row, col) = src.at<Vec3b>(row, col);
            }
            */

            // src ，通过指针去获取， 指针 -> Vec3b -> 获取
            uchar b1 = src.at<Vec3b>(row, col)[0];
            uchar g1 = src.at<Vec3b>(row, col)[1];
            uchar r1 = src.at<Vec3b>(row, col)[2];

            // blur_mat
            uchar b2 = blur_mat.at<Vec3b>(row, col)[0];
            uchar g2 = blur_mat.at<Vec3b>(row, col)[1];
            uchar r2 = blur_mat.at<Vec3b>(row, col)[2];

            // dst 254  1
            float k = mask_f.at<float>(row,col);

            dst.at<Vec3b>(row, col)[0] = b2*k + (1 - k)*b1;
            dst.at<Vec3b>(row, col)[1] = g2*k + (1 - k)*g1;
            dst.at<Vec3b>(row, col)[2] = r2*k + (1 - k)*r1;
        }
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_rocky_opencv2_Imgproc_00024Companion_nFilter2D(JNIEnv *env, jobject thiz,
                                                        jlong srcPtr, jlong dstPtr,
                                                        jlong kernelPtr) {
    Mat *src = reinterpret_cast<Mat *>(srcPtr);
    Mat *dst = reinterpret_cast<Mat *>(dstPtr);
    Mat *kernel = reinterpret_cast<Mat *>(kernelPtr);

    /**
     * filter2d(src, dst, kernel);
     */
    //降噪======================
    /**
     *  blur(src, dst);
     */

    //===================


//美颜
    /**
     *  equalize(src, dst);
     */

//彩色回放均衡

//转化通道
    /**
     *     crcb(src, dst);
     */
//使用HSV 改变亮度
    /**
     *   HSV(src, dst);
     */

    //remap
    /**
     * backProject(src, dst);
     */

    //直方均衡改变亮度
    /**
     *  hist_2(src, dst);
     */
    //查找 轮廓  从结果看 不是理想的 先留着疑问？？？？？？？？？？
    /**
     * findContoursss(src, dst);
     */

    //手写 亮度增强
    /**
     *   hsv_v(src, dst);
     */

    //手写 直方图
    /**
     *   Mat grey;
      cvtColor(*src, grey, CV_BGRA2GRAY);
      mEqualizeHist(grey, *dst);
     */

    //美容实现
    Mat mat;
    cvtColor(*src, mat, CV_BGRA2BGR);
//    bilateralFilter(mat,*dst,0,100,25);
    int size = 15;//模糊直径
    fatsBilateralBlur(mat, *dst, size);

    Mat skinMask;
    skinDetect(*src, skinMask);
    Mat fuseDst;
    fuseSkin(*src, *dst, fuseDst, skinMask);

    // 边缘的提升 (可有可无)
    Mat cannyMask;
    Canny(*src, cannyMask, 150, 300, 3, false);
    // & 运算  0 ，255
    bitwise_and(*src, *src, fuseDst, cannyMask);
    // 稍微提升一下对比度(亮度)
    add(fuseDst, Scalar(10, 10, 10), fuseDst);
    *dst = fuseDst;
}

