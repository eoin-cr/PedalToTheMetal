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
                SensorManager.SENSOR_DELAY_NORMAL);


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
//            previousAccelerations.add(new Float[]{x, y, z, Float.valueOf(System.currentTimeMillis())});

            xCoor.setText("X: "+x);
            yCoor.setText("Y: "+y);
            zCoor.setText("Z: "+z);
//            previousAccelerationString += Arrays.toString(new Float[]{x, y, z, (float) System.currentTimeMillis()});
            previousAccelerationString += x + "," + y + "," + z + "," + (float) System.currentTimeMillis() + "\n";
//            previousAccelerationsTextView.setText(previousAccelerationString);

        }
    }

    private void writeToFileAndResetString(String data, Context context, String emotion) {
        try {
//            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(context.openFileOutput("config.txt", Context.MODE_WORLD_READABLE));
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(context.openFileOutput(count++ + "-" + emotion + ".csv", 0));
            outputStreamWriter.write(data + "\n");
            outputStreamWriter.close();
            previousAccelerationString = "";
        }
        catch (IOException e) {
            Log.e("Exception", "File write failed: " + e.toString());
        }
    }

}
