//
// Created by rocky on 2022/7/14.
//

#include "bitmap_util.h"
#include "opencv.hpp"

//public static Bitmap createBitmap(int width, int height, @NonNull Config config)

//Bitmap.createBitmap(1080,1920,Bitmap.Config.ARGB_8888)

jobject bitmap_util::create_bitmap(JNIEnv *env, int width, int height, int type) {
    const char *bitmap_class_name = "android/graphics/Bitmap";
    jclass bitmap_class = env->FindClass(bitmap_class_name);
    jmethodID bitmap_mid = env->GetStaticMethodID(bitmap_class, "createBitmap",
                                                  "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");

    const char *config_name;
    //根据type 创建不同的config 这里只处理一种
    if (type == CV_8UC4) {
        config_name = "ARGB_8888";
    }

    const char *bitmap_config_class_name = "android/graphics/Bitmap$Config";
    //val config:Bitmap.Config = Bitmap.Config.valueOf("");
    jclass config_class = env->FindClass(bitmap_config_class_name);
    jmethodID jmethodId = env->GetStaticMethodID(config_class, "valueOf",
                                                 "(Ljava/lang/String;)Landroid/graphics/Bitmap$Config;");
    jstring type_name = env->NewStringUTF(config_name);
    jobject config_obj = env->CallStaticObjectMethod(config_class, jmethodId, type_name);
    jobject bitmap = env->CallStaticObjectMethod(bitmap_class, bitmap_mid, width, height,
                                                 config_obj);
     env->DeleteLocalRef(type_name);

    return bitmap;
}
