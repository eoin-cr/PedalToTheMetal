package com.example.pedaltothemetal;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

import com.example.accelerometerstorer.R;

public class PedalToTheMetal extends Activity {

    TextView emotion; // declare mood object

    @Override
    public void onCreate(Bundle savedInstanceState){

        super.onCreate(savedInstanceState);
        setTheme(R.style.Theme_PedalToTheMetal);
        setContentView(R.layout.main);


        emotion =(TextView)findViewById(R.id.emotion); // create mood object

        emotion.setText("Dying");
    }

}
