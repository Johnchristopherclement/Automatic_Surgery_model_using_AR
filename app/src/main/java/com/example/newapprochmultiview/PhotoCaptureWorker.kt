package com.example.newapprochmultiview


import android.bluetooth.BluetoothSocket
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import androidx.work.Worker
import androidx.work.WorkerParameters
import java.io.ByteArrayOutputStream
import java.io.IOException

class PhotoCaptureWorker(
    appContext: android.content.Context,
    workerParams: WorkerParameters
) : Worker(appContext, workerParams) {

    private var bluetoothSocket: BluetoothSocket? = null

    override fun doWork(): Result {
        bluetoothSocket = (applicationContext as BluetoothService).bluetoothSocket

        // Take photo and transmit via Bluetooth
        takePhotoAndTransmit()

        return Result.success()
    }

    private fun takePhotoAndTransmit() {
        try {
            val bitmap = takePhoto()
            transmitViaBluetooth(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error during photo capture and transmission: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun takePhoto(): Bitmap {
        // Implement your photo capture logic here
        // For simplicity, return a placeholder bitmap
        return BitmapFactory.decodeResource(applicationContext.resources, R.drawable.placeholder)
    }

    private fun transmitViaBluetooth(bitmap: Bitmap) {
        try {
            val stream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)

            val byteFormat = stream.toByteArray()
            val encodedData = Base64.encodeToString(byteFormat, Base64.NO_WRAP)

            val outputStream = bluetoothSocket!!.outputStream
            outputStream.write(encodedData.toByteArray())
            outputStream.write("stop".toByteArray())
            outputStream.flush()

            Log.d(TAG, "Photo transmitted successfully")
        } catch (e: IOException) {
            Log.e(TAG, "Error transmitting photo via Bluetooth: ${e.message}")
            e.printStackTrace()
        }
    }

    companion object {
        private const val TAG = "PhotoCaptureWorker"
    }
}
