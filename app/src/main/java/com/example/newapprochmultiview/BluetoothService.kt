package com.example.newapprochmultiview

import android.app.Service
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.compose.ui.unit.Constraints
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.work.*
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit

class BluetoothService : LifecycleService() {

    private var bluetoothAdapter: BluetoothAdapter? = null
    private var bluetoothDevice: BluetoothDevice? = null
    private var bluetoothSocket: BluetoothSocket? = null

    private val dateFormat = SimpleDateFormat("ss", Locale.getDefault())

    override fun onCreate() {
        super.onCreate()
        bluetoothAdapter = BluetoothAdapter.getDefaultAdapter()
        if (bluetoothAdapter == null || !bluetoothAdapter!!.isEnabled) {
            stopSelf()
            return
        }
        startForegroundService()
        connectToBluetoothDevice()
        startPhotoCaptureWorker()
    }

    override fun onDestroy() {
        super.onDestroy()
        disconnectBluetooth()
        stopForeground(true)
    }

    private fun startForegroundService() {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Bluetooth Service")
            .setContentText("Running")
            .setSmallIcon(R.drawable.ic_notification)
            .build()
        startForeground(NOTIFICATION_ID, notification)
    }

    private fun connectToBluetoothDevice() {
        val deviceName = "11844-SENSE2"
        bluetoothDevice = findDeviceByName(deviceName)
        if (bluetoothDevice == null) {
            Log.e(TAG, "Bluetooth device $deviceName not found")
            return
        }
        val uuid = UUID.fromString("94f39d29-7d6d-437d-973b-fba39e49d4ee")
        bluetoothSocket = bluetoothDevice!!.createRfcommSocketToServiceRecord(uuid)
        try {
            bluetoothSocket!!.connect()
        } catch (e: IOException) {
            Log.e(TAG, "Error connecting to Bluetooth device: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun findDeviceByName(deviceName: String): BluetoothDevice? {
        val pairedDevices = bluetoothAdapter!!.bondedDevices
        return pairedDevices.find { it.name == deviceName }
    }

    private fun disconnectBluetooth() {
        try {
            bluetoothSocket?.close()
        } catch (e: IOException) {
            Log.e(TAG, "Error closing Bluetooth socket: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun startPhotoCaptureWorker() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val photoCaptureRequest = PeriodicWorkRequestBuilder<PhotoCaptureWorker>(2, TimeUnit.SECONDS)
            .setConstraints(constraints)
            .build()

        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            WORKER_TAG,
            ExistingPeriodicWorkPolicy.REPLACE,
            photoCaptureRequest
        )
    }

    companion object {
        private const val TAG = "BluetoothService"
        private const val CHANNEL_ID = "BluetoothServiceChannel"
        private const val NOTIFICATION_ID = 1
        private const val WORKER_TAG = "PhotoCaptureWorker"
    }

    override fun onBind(intent: Intent): IBinder? {
        return super.onBind(intent)
    }
}
