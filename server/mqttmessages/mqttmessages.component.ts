import { Component, OnInit, Input } from '@angular/core';
import { MQTTmessage } from '../MQTTmessage';

@Component({
  selector: 'app-mqttmessages',
  templateUrl: './mqttmessages.component.html',
  styleUrls: ['./mqttmessages.component.css']
})
export class MQTTmessagesComponent implements OnInit {
  @Input() MQTTlog: MQTTmessage[] = [];

  constructor() { }

  ngOnInit(): void {
  }

}
