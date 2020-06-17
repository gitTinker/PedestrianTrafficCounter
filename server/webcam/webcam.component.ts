// https://stackoverflow.com/questions/37904023/how-to-use-webrtc-in-angular-2/48479628
// may want to use this https://stackoverflow.com/questions/59916165/how-to-do-video-chat-using-webrtc-in-angular8
// https://www.npmjs.com/package/webrtc-adapter

import { Component, OnInit, ViewChild } from '@angular/core';
//import  adapter  from 'webrtc-adapter';

@Component({
  selector: 'app-webcam',
  templateUrl: './webcam.component.html',
  styleUrls: ['./webcam.component.css']
})
export class WebcamComponent implements OnInit {

  //@ViewChild('hardwareVideo') hardwareVideo: any;
  //_navigator = <any> navigator;
  //localStream;

  constructor() { }

  ngOnInit(): void {
    //const video = this.hardwareVideo.nativeElement;
    //this._navigator = <any>navigator;
    //this._navigator.getUserMedia = ( this._navigator.getUserMedia || this._navigator.webkitGetUserMedia
      //|| this._navigator.mozGetUserMedia || this._navigator.msGetUserMedia );

      //this._navigator.mediaDevices.getUserMedia({video: true})
        //.then((stream) => {
          //this.localStream = stream;
          //video.src = window.URL.createObjectURL(stream);
          //video.play();
      //});
  }

  // stopStream() {
  //   const tracks = this.localStream.getTracks();
  //   tracks.forEach((track) => {
  //     track.stop();
  //   });
  // }

}
