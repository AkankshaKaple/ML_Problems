import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatBoxComponent } from './chat-box.component';
import { BrowserModule, By } from '@angular/platform-browser';
import { MatInputModule, MatCardModule, MatIconModule, MatToolbarModule, MatDividerModule, MAT_CHECKBOX_CLICK_ACTION } from '@angular/material';
import { RouterTestingModule } from '@angular/router/testing';
import { DebugElement } from '@angular/core';
import {FormsModule, ReactiveFormsModule} from '@angular/forms';



describe('ChatBoxComponent', () => {
  let component: ChatBoxComponent;
  let fixture: ComponentFixture<ChatBoxComponent>;
  let de: DebugElement;
  let el: HTMLElement;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ChatBoxComponent ],

      imports: [
        BrowserModule,
        MatInputModule,
        MatCardModule,
        MatIconModule,
        MatToolbarModule,
        FormsModule,
        MatDividerModule,
        ReactiveFormsModule,
        RouterTestingModule.withRoutes([{path: '', component:ChatBoxComponent}])
      ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ChatBoxComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  
  it('it should push message in array',() =>{
    console.log(component.newMsg.setValue('pagal'));
    console.log(component.newMsg.value);
    component.sendMsg();
    console.log(component.send);
   
  });


  it('should call the sendMsg method', async(()=>{
    fixture.detectChanges()
    spyOn(component,'sendMsg');
    el: fixture.debugElement.query(By.css('button')).nativeElement;

  }))

  // it('should return the sendMsg method', async(()=>{
  //   fixture.detectChanges()
  //   spyOn(component, 'sendMsg');
    
    

  // }))

  it('should not accept whitespace',async(()=>{
    component.newMsg.setValue('                      ');
    component.sendMsg();
    expect(component.send.length==0).toBeTruthy();
  }))
   
  it('should accept whitespace',async(()=>{
    component.newMsg.setValue('                      ');
    component.sendMsg();
    expect(component.send.length!=0).toBeFalsy();
  }))
  
  it('should set submitted to true', async(()=>{
    expect(component.onKeyup).toBeTruthy();
  }))
  
   it('should call the onkeyup method', async(()=>{
     fixture.detectChanges();
     spyOn(component,'onKeyup');
     el: fixture.debugElement.query(By.css('button')).nativeElement;
    //  el.enter();
     expect(component.onKeyup).toHaveBeenCalledTimes(0);
   }))

   it('should not show the chatbox', async(()=>{
     expect(component.toggle()).toBeFalsy();
   }))


});
 
