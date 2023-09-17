package com.example.pedaltothemetal;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.StrictMode;
import android.widget.TextView;
import android.widget.Toast;

//import org.apache.http.HttpEntity;
//import org.apache.http.HttpResponse;
//import org.apache.http.client.HttpClient;
//import org.apache.http.client.methods.HttpPost;
//import org.apache.http.entity.StringEntity;
//import org.apache.http.impl.client.HttpClients;
//import org.apache.http.util.EntityUtils;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import com.example.accelerometerstorer.R;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;

public class PedalToTheMetal extends Activity implements SensorEventListener {
    TextView emotion; // declare mood object
    TextView playlist;
    private SensorManager sensorManager;
    String previousAccelerationString = "";
    String header = "x_acc,y_acc,z_acc,time";
    Map<Integer, String> dictionary = new HashMap<>();
    OkHttpClient client = new OkHttpClient();

    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setTheme(R.style.Theme_PedalToTheMetal);
        setContentView(R.layout.main);

        StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
        StrictMode.setThreadPolicy(policy);

        dictionary.put(0, "Happy");
        dictionary.put(1, "Sad");
        dictionary.put(2, "Chill");
        dictionary.put(3, "Angry");
        dictionary.put(4, "Invalid");

        emotion = (TextView) findViewById(R.id.emotion); // create mood object
        playlist = (TextView) findViewById(R.id.Playlist);

        emotion.setText("Currently loading");

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        // add listener. The listener will be  (this) class
        sensorManager.registerListener(this,
                sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION),
                50000);

        Runnable queryRunnable = () -> queryEmotion();

        ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
        executor.scheduleAtFixedRate(queryRunnable, 3, 3, TimeUnit.SECONDS);
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
//        try {
//            HttpClient httpClient = HttpClients.createDefault();
//            HttpPost httpPost = new HttpPost("http://10.33.134.164:5000/post");
//            httpPost.setHeader("Content-Type", "text/json");
//            String requestBody = "{\"csv_as_str\": \"" + header + "\n" + previousAccelerationString + "\"}";
//            StringEntity requestEntity = new StringEntity(requestBody);
//
//            // Set the request entity
//            httpPost.setEntity(requestEntity);
//
//            // Execute the request and get the response
//            HttpResponse response = httpClient.execute(httpPost);
//
//            // Get the response entity
//            HttpEntity entity = response.getEntity();
//            if (entity != null) {
//                String responseContent = EntityUtils.toString(entity);
//                System.out.println("Response: " + responseContent);
//            }
//        } catch (Exception e) {
//
//        }

//        try {
//            URL url = new URL("http://10.33.134.164:5000/post");
//            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
//            connection.setRequestMethod("POST");
//            connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
//            connection.setDoOutput(true);
//            String requestBody = "{\"csv_as_str\": \"" + header + "\n" + previousAccelerationString + "\"}";
//            previousAccelerationString = "";
//
//            // Get the output stream from the connection
//            try (OutputStream os = connection.getOutputStream()) {
//                byte[] input = requestBody.getBytes("utf-8");
//                os.write(input, 0, input.length);
//            }
//            int responseCode = connection.getResponseCode();
//            try (BufferedReader reader = new BufferedReader(
//                    new InputStreamReader(connection.getInputStream()))) {
//                String line;
//                StringBuilder response = new StringBuilder();
//
//                while ((line = reader.readLine()) != null) {
//                    response.append(line);
//                }
//                String[] lines = response.toString().split("\n");
//                int emoNum = Integer.valueOf(lines[0]);
//                emotion.setText(dictionary.get(emoNum));
//            }
//            connection.disconnect();
//        } catch (Exception e) {
//            Toast.makeText(getApplicationContext(), "Exception" + e, Toast.LENGTH_LONG).show();
//        }

        try {
            RequestBody formBody = new FormBody.Builder()
                    .add("csv_as_str", header + "\n" + previousAccelerationString + "\n")
                    .build();
            Request request = new Request.Builder()
                    .url("http://10.33.134.164:5000/post")
                    .post(formBody)
                    .build();
            previousAccelerationString = "";

            Call call = client.newCall(request);
            Response response = call.execute();

            String[] lines = response.body().string().split("\n");
                int emoNum = Integer.valueOf(lines[0]);
                emotion.setText(dictionary.get(emoNum));
                playlist.setText(String.join("\n", Arrays.copyOfRange(lines, 1, 6)));
                response.close();

        } catch (Exception e) {

        }
    }

}