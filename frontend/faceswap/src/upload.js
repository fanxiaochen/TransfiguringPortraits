
import axios from 'axios';



//export const serverUrl = 'http://127.0.0.1:5000';
//export const serverUrl = 'http://192.168.31.126:5000';
export const serverUrl = 'http://xiaochenfan.xyz:9080';
export const http = axios.create({
  baseURL: serverUrl
});

export const uuidGen = require('react-native-uuid');

export const submitFormData = function(imagePath, item, uuid) {
    let body = new FormData();
    //imagePath = '/home/xiaochen/Workspace/TransfiguringPortraits';
    imageFile = imagePath.substr(imagePath.lastIndexOf('/')+1)
    imageName = imageFile.substr(0, imageFile.lastIndexOf('.'))
    console.log(imageFile)
    console.log(imageName)
    //body.append('photo', {uri: imagePath,name: 'xi1',filename :'xi1.png',type: 'image/png'});
    body.append('image', {uri: imagePath,name: imageName,filename :imageFile,type: 'image/jpeg'});
    body.append('item', item);
    body.append('uuid', uuid);
    //body.append('Content-Type', 'image/jpeg');

    let config = {
      headers:{'Content-Type':'multipart/form-data'}
    }

    http.post('/', body, config)
    .then((res) => { console.log("response" +JSON.stringify(res)); })
    .catch((e) => console.log(e))
    .done()
  }

export const waitingForImage = ()=>{
  http.get('/result')
  .then((response) => {
      //check if status is completed, if it is stop polling 
      if(response.data.status = 'completed') {
            clearInterval(this.pollInterval) //won't be polled anymore 
      }
      this.status = response; 
    });
}

export const mounted = ()=>{
  //check if the status is completed, if not fetch data every 10minutes
  if(this.status.status != 'completed') {
    this.pollInterval = setInterval(waitingForImage, 2000) //save reference to the interval
    setTimeout(() => {clearInterval(this.pollInterval)}, 36000000) //stop polling after an hour
  }
}


export const formImageBody = function(imagePath) {
    let body = new FormData();
    //imagePath = '/home/xiaochen/Workspace/TransfiguringPortraits';
    imageFile = imagePath.substr(imagePath.lastIndexOf('/')+1)
    imageName = imageFile.substr(0, imageFile.lastIndexOf('.'))
    console.log(imageFile)
    console.log(imageName)
    //body.append('photo', {uri: imagePath,name: 'xi1',filename :'xi1.png',type: 'image/png'});
    body.append('image', {uri: imagePath,name: imageName,filename :imageFile,type: 'image/jpeg'});
    body.append('Content-Type', 'image/jpeg');

    fetch(serverUrl+'/image',{ method: 'POST',headers:{  
        "Content-Type": "multipart/form-data",
        "otherHeader": "foo",
        } , body :body} )
      .then((res) => res.json())
      .then((res) => { console.log("response" +JSON.stringify(res)); })
      .catch((e) => console.log(e))
      .done()
  }