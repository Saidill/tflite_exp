package com.example.test_tflite

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.test_tflite.ml.ModelCoba
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var camera: Button
    private lateinit var gallery: Button
    private lateinit var imageView: ImageView
    private lateinit var result: TextView
    private val imageSize = 150

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.layout)

        gallery = findViewById(R.id.button2)
        result = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)


        gallery.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent, 1)
        }
    }

    private fun classifyImage(image: Bitmap) {
        try {
            val model = ModelCoba.newInstance(applicationContext)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 150, 150, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            var pixel = 0
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val value = intValues[pixel++]
                    byteBuffer.putFloat(((value shr 16) and 0xFF) * (1f / 255))
                    byteBuffer.putFloat(((value shr 8) and 0xFF) * (1f / 255))
                    byteBuffer.putFloat((value and 0xFF) * (1f / 255))
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            val confidences = outputFeature0.floatArray
            val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: -1
            val maxConfidence = confidences[maxPos] * 100 // Konversi ke persentase
            val classes = arrayOf(
                "baca", "bantu", "makan", "bapak", "buangairkecil", "buat", "halo",
                "ibu", "maaf", "mau", "kamu", "nama", "pagi", "paham", "sakit",
                "sama-sama", "saya", "selamat", "siapa", "tanya", "tempat",
                "terimakasih", "terlambat", "tidak", "tolong", "tugas"
            )

            // Tampilkan hasil dengan confidence
            result.text = "Prediksi: ${classes[maxPos]} \nAkurasi: %.2f%%".format(maxConfidence)

            model.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                val image = data?.extras?.get("data") as Bitmap?
                image?.let {
                    val dimension = it.width.coerceAtMost(it.height)
                    val thumbnail = ThumbnailUtils.extractThumbnail(it, dimension, dimension)
                    imageView.setImageBitmap(thumbnail)

                    val scaledImage = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
                    classifyImage(scaledImage)
                }
            } else if (requestCode == 1) {
                val uri: Uri? = data?.data
                val image = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                imageView.setImageBitmap(image)

                val scaledImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                classifyImage(scaledImage)
            }
        }
    }
}
