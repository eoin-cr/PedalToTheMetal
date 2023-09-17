package com.example.pedaltothemetal

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Alignment.Companion.CenterHorizontally
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.accelerometerstorer.R
import com.example.pedaltothemetal.ui.theme.AccelerometerStorerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AccelerometerStorerTheme() {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Row(
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .fillMaxHeight(0.15f)
                            .padding(top = 10.dp)
                    ) {
                        Image(
                            painter = painterResource(id = R.drawable.logo_htn),
                            contentDescription = "HTN logo"
                        )
                        Spacer(modifier = Modifier.width(100.dp))
                        Image(
                            painter = painterResource(id = R.drawable.logo_2),
                            contentDescription = "Logo 2"
                        )
                    }
                    Spacer(modifier = Modifier.height(20.dp))
                    Mood("Happy", Modifier.align(CenterHorizontally))
                }
            }
        }
    }
}

@Composable
fun Mood(mood: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Text(
            text = stringResource(id = R.string.mood_is),
            fontSize = 32.sp
        )
        Text(
            text = "$mood",
            fontSize = 48.sp,
            modifier = Modifier.align(CenterHorizontally)
        )
    }
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    AccelerometerStorerTheme {
        Column {
            Mood("Happy")
            Image(
                painter = painterResource(R.drawable.logo_1),
                contentDescription = "Logo 1"
            )
        }
    }
}