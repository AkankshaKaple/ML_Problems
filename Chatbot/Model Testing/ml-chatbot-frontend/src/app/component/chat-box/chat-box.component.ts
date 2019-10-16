import { Component, OnInit } from '@angular/core';
import { FormControl } from '@angular/forms';
import { AiService } from '../../services/ai.service';

@Component({
  selector: 'app-chat-box',
  templateUrl: './chat-box.component.html',
  styleUrls: ['./chat-box.component.scss']
})
export class ChatBoxComponent implements OnInit {

  message: any;
  type: any;
  date: any;
  backendData: any;
  usermessage: string;
  newMsg = new FormControl('');

  public show = false;

  send: Array<Object> = [

  ];

  constructor(private http: AiService) { }

  ngOnInit() {

  }


  toggle() {
    this.show = !this.show;
  }

  sendMsg() {
    let d = new Date();
    if (this.newMsg.value.trim().length > 0) {
      this.send.push({ message: this.newMsg.value, type: 'S', date: d });


      this.usermessage = this.newMsg.value;
      this.http.textConversation(this.usermessage).subscribe(data => {
        this.backendData = data;
        if (this.backendData.data.answer.includes('http')) {
          let templink: any;
          let tempdata: any;
          templink = this.backendData.data.answer.substring(this.backendData.data.answer.indexOf('http'), this.backendData.data.length);
          tempdata = this.backendData.data.answer.substring(0, this.backendData.data.answer.indexOf('http'));
          this.send.push({ message: tempdata, link: templink, type: 'L', date: d });

        } else {
          this.send.push({ message: this.backendData.data.answer, type: 'R', date: d });
          localStorage.setItem('user_id', this.backendData.data.user_id);
        }
      });
      // }

      this.newMsg.reset();
    } else {
      this.newMsg.reset();
    }

  }

  onKeyup(event) {
    console.log(event);
  }

}
