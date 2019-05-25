
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
    imageFile = imagePath.substr(imagePath.lastIndexOf('/')+1)
    imageName = imageFile.substr(0, imageFile.lastIndexOf('.'))
    console.log(imageFile)
    console.log(imageName)
    body.append('image', {uri: imagePath,name: imageName,filename :imageFile,type: 'image/jpeg'});
    body.append('item', item);
    body.append('uuid', uuid);

    let config = {
      headers:{'Content-Type':'multipart/form-data'}
    }

    http.post('/', body, config)
    .then((res) => { console.log("response" +JSON.stringify(res)); })
    .catch((e) => console.log(e))
    .done()

  }