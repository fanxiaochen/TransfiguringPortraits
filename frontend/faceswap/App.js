/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 * @flow
 */

import React, {Component} from 'react';
//import {StyleSheet, Text, View, Button} from 'react-native';
//import axios from 'axios';
//import ImagePicker from 'react-native-image-picker';



import AppContainer from './src/index'

//import Welcome from './src/welcome'
//
//
//const AppNavigator = createStackNavigator({
//    Home: {
//      screen: Welcome
//    }
//  });

//const serverUrl = 'http://localhost:5000';
//const http = axios.create({
//  baseURL: serverUrl
//});
//
//const options = {
//  title: 'Select Avatar',
//  customButtons: [{ name: 'fb', title: 'Choose Photo from Facebook' }],
//  storageOptions: {
//    skipBackup: true,
//    path: 'images',
//  },
//};
//
//// open camera
//ImagePicker.launchCamera(options, (response) => {
//  if (response.didCancel) {
//    console.log('User cancelled image picker');
//  } else if (response.error) {
//    console.log('ImagePicker Error: ', response.error);
//  } else if (response.customButton) {
//    console.log('User tapped custom button: ', response.customButton);
//  } else {
//    const source = { uri: response.uri };
// 
//    // You can also display the image using data:
//    // const source = { uri: 'data:image/jpeg;base64,' + response.data };
// 
//    this.setState({
//      avatarSource: source,
//    });
//  }
//});

export default class App extends Component {

 // onImageSend(){
 //   let body = new FormData();
 //   imagePath = '/home/xiaochen/Workspace/TransfiguringPortraits';
 //   body.append('photo', {uri: imagePath,name: 'xi1',filename :'xi1.png',type: 'image/png'});
 //   body.append('Content-Type', 'image/png');

 //   fetch(serverUrl+'/image',{ method: 'POST',headers:{  
 //       "Content-Type": "multipart/form-data",
 //       "otherHeader": "foo",
 //       } , body :body} )
 //     .then((res) => checkStatus(res))
 //     .then((res) => res.json())
 //     .then((res) => { console.log("response" +JSON.stringify(res)); })
 //     .catch((e) => console.log(e))
 //     .done()
 // }

 // onItemSend(){
 //   http.post('/item',{
 //     // image
 //     // item
 //   });

 // }

 // onMessageSend(){
 //   http.post('/',{
 //     // image
 //     // item
 //   });
 // }

 // _onPressButton(){

 // }

  render() {
    return (
      <AppContainer/>
    //  <View style={styles.container}>
    //    <Text style={styles.welcome}>A simple demo of face swapping</Text>
    //    <Button
    //        onPress={this._onPressButton}
    //        title="Let's start"
    //    />
    //  </View>
    );
  }
}

//const styles = StyleSheet.create({
//  container: {
//    flex: 1,
//    justifyContent: 'center',
//    alignItems: 'center',
//    backgroundColor: '#F5FCFF',
//  },
//  welcome: {
//    fontSize: 20,
//    textAlign: 'center',
//    margin: 10,
//  },
//});
