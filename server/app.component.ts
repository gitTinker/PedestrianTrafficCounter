import { Component, OnInit, ViewChild, ElementRef, OnDestroy } from '@angular/core';
import { Subscription } from 'rxjs';
import { IMqttMessage, MqttService } from 'ngx-mqtt';
import { MQTTmessage } from './MQTTmessage';
import { TrafficResults } from './TrafficResults';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'Pedestrian Traffic';
  private subscription: Subscription;
  topicname: any = "TrafficCounter/#";
  msg: any = "";
  direction: any = false;
  isConnected: boolean = false;
  @ViewChild('msglog', { static: true }) msglog: ElementRef;
  messageQueue: MQTTmessage[] =[];
  trafficResults: TrafficResults = {topic: "", status: "", quantityLeft: 0, quantityRight: 0, occupancy: 0};

  constructor(private _mqttService: MqttService) { }

    ngOnInit(): void {
      //var incomingMsg: MQTTmessage = {topic: "app/le", msg: "04r"}
      //var length = this.messageQueue.push(incomingMsg);
    }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }


  subscribeNewTopic(): void {
    console.log('inside subscribe new topic')
    this.subscription = this._mqttService.observe(this.topicname).subscribe((message: IMqttMessage) => {
      //this.msg = message;
      console.log('msg: ', message)
      //this.logMsg('Message: ' + message.payload.toString() + '<br> for topic: ' + message.topic);
      var incomingMsg: MQTTmessage = {topic: message.topic, msg: message.payload.toString()}
      if (this.messageQueue.length > 21){
        // too many entries in the array, remove the oldest 5 entries
        this.messageQueue = this.messageQueue.slice(1)
      }
      var length = this.messageQueue.push(incomingMsg);
      if (message.topic == "TrafficCounter/TrafficDetected/Status") {this.trafficResults.status = message.payload.toString()};
      if (message.topic == "TrafficCounter/TrafficDetected/Left") {
        this.trafficResults.quantityLeft = Number(message.payload.toString());
        this.trafficResults.occupancy = this.direction ? this.trafficResults.quantityLeft - this.trafficResults.quantityRight : this.trafficResults.quantityRight - this.trafficResults.quantityLeft;
      };
      if (message.topic == "TrafficCounter/TrafficDetected/Right") {
        this.trafficResults.quantityRight = Number(message.payload.toString())};
        this.trafficResults.occupancy = this.direction ? this.trafficResults.quantityLeft - this.trafficResults.quantityRight : this.trafficResults.quantityRight - this.trafficResults.quantityLeft;
    });
    //this.logMsg('subscribed to topic: ' + this.topicname);
    this.trafficResults.topic = this.topicname;
    var incomingMsg: MQTTmessage = {topic: "subscribed to topic: ", msg: this.topicname}
    var length = this.messageQueue.push(incomingMsg);
}

  sendmsg(): void {
    // use unsafe publish for non-ssl websockets
    this._mqttService.unsafePublish(this.topicname, this.msg, { qos: 1, retain: true })
    this.msg = ''
  }

  logMsg(message): void {
    this.msglog.nativeElement.innerHTML += '<br><hr>' + message;
  }

  // clear(): void {
  //   this.msglog.nativeElement.innerHTML = '';
  // }

}
