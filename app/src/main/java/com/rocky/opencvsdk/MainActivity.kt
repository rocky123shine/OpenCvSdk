package com.rocky.opencvsdk

import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import com.rocky.opencvsdk.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val imageView: ImageView = findViewById(R.id.image)
        val imageView1: ImageView = findViewById(R.id.image1)

        var bitmap = BitmapFactory.decodeResource(resources, R.mipmap.img_1)
        imageView.setImageBitmap(bitmap)
//        bitmap = NDKBitmapUtils.warpAffine(bitmap)
//        imageView1.setImageBitmap(bitmap)
        //java层封装opencv
        imageView1.setImageBitmap(OpencvUtils.mask(bitmap))


    }
    companion object {
        // Used to load the 'opencvsdk' library on application startup.
        init {
            System.loadLibrary("mopencv")
        }
    }
}