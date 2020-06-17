import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { MQTTmessagesComponent } from './mqttmessages.component';

describe('MQTTmessagesComponent', () => {
  let component: MQTTmessagesComponent;
  let fixture: ComponentFixture<MQTTmessagesComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ MQTTmessagesComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(MQTTmessagesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
