package com.example.krishi4

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.krishi4.ml.LiteMode
import com.example.krishi4.ui.theme.Krishi4Theme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : ComponentActivity() {

    lateinit var selectBtn: Button
    lateinit var predBtn: Button
    lateinit var resView: TextView
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectBtn = findViewById(R.id.selectBtn)
        predBtn = findViewById(R.id.predictBtn)
        resView = findViewById(R.id.resView)
        imageView = findViewById(R.id.imageView)

        var labels = application.assets.open("labels.txt").bufferedReader().readLines()

        //image processor
        var imageProcessor=ImageProcessor.Builder()
            .add(ResizeOp(256,256, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        selectBtn.setOnClickListener {
            var intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/")
            startActivityForResult(intent, 100)

        }

        predBtn.setOnClickListener {

            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)

            tensorImage = imageProcessor.process(tensorImage)
            val model = LiteMode.newInstance(this)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var maxIdx = 0;
            outputFeature0.forEachIndexed{index, fl ->
                if(outputFeature0[maxIdx]<fl){
                    maxIdx=index
                }
            }

            resView.setText(labels[maxIdx])
            model.close()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode==100){
            var uri = data?.data;
            bitmap= MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
            imageView.setImageBitmap(bitmap)
        }
    }
}