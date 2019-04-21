
import axios from 'axios';


//export const serverUrl = 'http://127.0.0.1:5000';
export const serverUrl = 'http://192.168.31.126:5000';
export const http = axios.create({
  baseURL: serverUrl
});


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