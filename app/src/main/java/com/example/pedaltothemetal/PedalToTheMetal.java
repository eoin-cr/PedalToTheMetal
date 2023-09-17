package com.example.pedaltothemetal;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import com.example.accelerometerstorer.R;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class PedalToTheMetal extends Activity implements SensorEventListener {
    TextView emotion; // declare mood object
    private SensorManager sensorManager;
    String previousAccelerationString = "";
    String header = "x_acc,y_acc,z_acc,time";
    Map<Integer, String> dictionary = new HashMap<>();

    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setTheme(R.style.Theme_PedalToTheMetal);
        setContentView(R.layout.main);

        Runnable queryRunnable = () -> queryEmotion();

        ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
        executor.scheduleAtFixedRate(queryRunnable, 0, 3, TimeUnit.SECONDS);

        dictionary.put(0, "Happy");
        dictionary.put(1, "Sad");
        dictionary.put(2, "Chill");
        dictionary.put(3, "Angry");
        dictionary.put(4, "Invalid");

        emotion = (TextView) findViewById(R.id.emotion); // create mood object

        emotion.setText("Currently loading");

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        // add listener. The listener will be  (this) class
        sensorManager.registerListener(this,
                sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION),
                50000);
        Toast.makeText(getApplicationContext(), "LoadA", Toast.LENGTH_SHORT).show();
        queryEmotion();
        Toast.makeText(getApplicationContext(), "LoadB", Toast.LENGTH_SHORT).show();
    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    public void onSensorChanged(SensorEvent event) {

        // check sensor type
        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {

            // assign directions
            float x = event.values[0];
            float y = event.values[1];
            float z = event.values[2];
            previousAccelerationString += x + "," + y + "," + z + "," + System.currentTimeMillis() + "\n";

        }
    }

    public void queryEmotion() {
        try {
            URL url = new URL("http://10.33.134.164/post");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            String requestBody = "{\"csv-as-str\": \"" + header + "\n" + previousAccelerationString + "\"}";
            previousAccelerationString = "";

            // Get the output stream from the connection
            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = requestBody.getBytes("utf-8");
                os.write(input, 0, input.length);
            }
            int responseCode = connection.getResponseCode();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(connection.getInputStream()))) {
                String line;
                StringBuilder response = new StringBuilder();

                while ((line = reader.readLine()) != null) {
                    response.append(line);
                }
                String[] lines = response.toString().split("\n");
                int emoNum = Integer.valueOf(lines[0]);
                emotion.setText(dictionary.get(emoNum));
            }
        } catch (Exception e) {
            Toast.makeText(getApplicationContext(), "Exception" + e, Toast.LENGTH_LONG).show();
        }
    }

}