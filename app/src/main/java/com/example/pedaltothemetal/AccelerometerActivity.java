package com.example.pedaltothemetal;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Button;
import android.widget.Toast;

//import androidx.compose.material3.Button;

import com.example.accelerometerstorer.R;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;

public class AccelerometerActivity extends Activity implements SensorEventListener {
    private SensorManager sensorManager;

    TextView xCoor; // declare X axis object
    TextView yCoor; // declare Y axis object
    TextView zCoor; // declare Z axis object
//    List<Float[]> previousAccelerations = new ArrayList<>();
    String previousAccelerationString = "";
//    TextView previousAccelerationsTextView;
    static int count = 0;

    @Override
    public void onCreate(Bundle savedInstanceState){

        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        xCoor=(TextView)findViewById(R.id.xcoor); // create X axis object
        yCoor=(TextView)findViewById(R.id.ycoor); // create Y axis object
        zCoor=(TextView)findViewById(R.id.zcoor); // create Z axis object

        sensorManager=(SensorManager)getSystemService(SENSOR_SERVICE);
        // add listener. The listener will be  (this) class
        sensorManager.registerListener(this,
                sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION),
                50000);


        Button button = (Button)findViewById(R.id.happyButton);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // click handling code
                Toast.makeText(getApplicationContext(), "Happy button has been clicked!", Toast.LENGTH_LONG).show();
                writeToFileAndResetString(previousAccelerationString, getApplicationContext(), "happy");
            }
        });

        button = (Button)findViewById(R.id.sadButton);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // click handling code
                Toast.makeText(getApplicationContext(), "Sad button has been clicked!", Toast.LENGTH_LONG).show();
                writeToFileAndResetString(previousAccelerationString, getApplicationContext(), "sad");
            }
        });
        button = (Button)findViewById(R.id.angryButton);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // click handling code
                Toast.makeText(getApplicationContext(), "Angry button has been clicked!", Toast.LENGTH_LONG).show();
                writeToFileAndResetString(previousAccelerationString, getApplicationContext(), "angry");
            }
        });
        button = (Button)findViewById(R.id.chillButton);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // click handling code
                Toast.makeText(getApplicationContext(), "Chill button has been clicked!", Toast.LENGTH_LONG).show();
                writeToFileAndResetString(previousAccelerationString, getApplicationContext(), "chill");
            }
        });
        button = (Button)findViewById(R.id.invalidButton);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // click handling code
                Toast.makeText(getApplicationContext(), "Invalid button has been clicked!", Toast.LENGTH_LONG).show();
                writeToFileAndResetString(previousAccelerationString, getApplicationContext(), "invalid");
            }
        });
        button = (Button)findViewById(R.id.clearButton);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // click handling code
                Toast.makeText(getApplicationContext(), "Clear button has been clicked!", Toast.LENGTH_LONG).show();
                previousAccelerationString = "";
            }
        });

    }

    public void onAccuracyChanged(Sensor sensor,int accuracy){

    }

    public void onSensorChanged(SensorEvent event){

        // check sensor type
        if(event.sensor.getType()==Sensor.TYPE_LINEAR_ACCELERATION){

            // assign directions
            float x=event.values[0];
            float y=event.values[1];
            float z=event.values[2];

            xCoor.setText("X: "+x);
            yCoor.setText("Y: "+y);
            zCoor.setText("Z: "+z);
            previousAccelerationString += x + "," + y + "," + z + "," + System.currentTimeMillis() + "\n";

        }
    }

    private void writeToFileAndResetString(String data, Context context, String emotion) {
        try {
//            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(context.openFileOutput("config.txt", Context.MODE_WORLD_READABLE));
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(context.openFileOutput(count++ + "-" + emotion + ".csv", 0));
            String header = "x_acc,y_acc,z_acc,time";
            outputStreamWriter.write(header + "\n" + data + "\n");
            outputStreamWriter.close();
            previousAccelerationString = "";
        }
        catch (IOException e) {
            Log.e("Exception", "File write failed: " + e.toString());
        }
    }

}
