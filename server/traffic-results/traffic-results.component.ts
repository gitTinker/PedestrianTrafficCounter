import { Component, OnInit, Input } from '@angular/core';
import { TrafficResults } from '../TrafficResults';

@Component({
  selector: 'app-traffic-results',
  templateUrl: './traffic-results.component.html',
  styleUrls: ['./traffic-results.component.css']
})
export class TrafficResultsComponent implements OnInit {

  @Input() TrafficResults: TrafficResults;

  constructor() { }

  ngOnInit(): void {
  }

}
