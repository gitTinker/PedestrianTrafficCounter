//
// https://medium.com/@anant.lalchandani/dead-simple-mqtt-example-over-websockets-in-angular-b9fd5ff17b8e
//
// 1st:  ng new TrafficCounter    //create this project
// 2nd:  cd TrafficCounter
// 3rd:  npm i ngx-mqtt --save    // install dependancy
// 2020601 TIM

import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from "@angular/forms";
import { AppComponent } from './app.component';
import { MqttModule, IMqttServiceOptions } from "ngx-mqtt";
import { MQTTmessagesComponent } from './mqttmessages/mqttmessages.component';
import { TrafficResultsComponent } from './traffic-results/traffic-results.component';
import { WebcamComponent } from './webcam/webcam.component';
export const MQTT_SERVICE_OPTIONS: IMqttServiceOptions = {
  hostname: '10.0.3.139',
  port: 9001,
  path: ''
}

@NgModule({
  declarations: [
    AppComponent,
    MQTTmessagesComponent,
    TrafficResultsComponent,
    WebcamComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    MqttModule.forRoot(MQTT_SERVICE_OPTIONS)
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
