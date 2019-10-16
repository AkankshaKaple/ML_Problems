import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { DemoMaterialModule } from './material';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import {FlexLayoutModule} from '@angular/flex-layout';
import { ChatBoxComponent } from './component/chat-box/chat-box.component';
import {ScrollDispatchModule} from '@angular/cdk/scrolling';
import {MatIconModule} from '@angular/material/icon';
import {FormsModule, ReactiveFormsModule} from '@angular/forms';
import {AiService} from './services/ai.service';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';



@NgModule({
  declarations: [
    AppComponent,
    ChatBoxComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    DemoMaterialModule,
    BrowserAnimationsModule,
    FlexLayoutModule,
    MatIconModule,
    ScrollDispatchModule,
    FormsModule, ReactiveFormsModule,
    HttpClientModule
  ],
  providers: [AiService],
  bootstrap: [AppComponent]
})
export class AppModule { }
