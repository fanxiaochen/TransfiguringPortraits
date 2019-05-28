
import axios from 'axios';
import { Platform, CameraRoll } from 'react-native';
import RNFS from 'react-native-fs';

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

  export const download = function(uri) {
    if (!uri) return null;
    return new Promise((resolve, reject) => {
        let dirs = Platform.OS === 'ios' ? RNFS.LibraryDirectoryPath : RNFS.ExternalDirectoryPath; 
        const downloadDest = `${dirs}/${((Math.random() * 10000000) | 0)}.jpg`;
        const formUrl = uri;
        const options = {
            fromUrl: formUrl,
            toFile: downloadDest,
            background: true,
            begin: (res) => {
                console.log('begin', res);
                console.log('contentLength:', res.contentLength / 1024 / 1024, 'M');
            },
        };
        try {
            const ret = RNFS.downloadFile(options);
            ret.promise.then(res => {
                console.log('success', res);
                console.log('file://' + downloadDest)
                var promise = CameraRoll.saveToCameraRoll(downloadDest);
                promise.then(function(result) {
                    alert('Save successfully！Address：\n' + result);
                }).catch(function(error) {
                    console.log('error', error);
                    alert('Save failed！\n' + error);
                });
                resolve(res);
            }).catch(err => {
                reject(new Error(err))
            });
        } catch (e) {
            reject(new Error(e))
        }

    })

}