
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class AiService {
  baseUrl = 'http://0.0.0.0:5000';
  // baseUrl='http://192.168.0.116:5000/'
  data = '';
  constructor(private httpClient: HttpClient) {}

  textConversation(text: string) {
    return this.httpClient.get(this.baseUrl + '/ask/' + text);
  }

}
