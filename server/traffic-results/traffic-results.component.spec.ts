import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TrafficResultsComponent } from './traffic-results.component';

describe('TrafficResultsComponent', () => {
  let component: TrafficResultsComponent;
  let fixture: ComponentFixture<TrafficResultsComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TrafficResultsComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(TrafficResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
