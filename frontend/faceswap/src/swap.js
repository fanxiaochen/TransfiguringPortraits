import React, {Component} from 'react'
import {
    View,
    Button,
    Image,
    Alert,
    Dimensions,
    StyleSheet
} from 'react-native'

import {http, download} from './upload'

let screenWidth = Dimensions.get('window').width;
let screenHeight = Dimensions.get('window').height;

export default class Swap extends Component {
    constructor(props){
        super(props);
        this.state = {
          next_idx: 0,
          cur_url: '',
          height: 0,
          width: 0
        }

        this.receiveImage();

      }

    resizeImage() {
      Image.getSize(this.state.cur_url, (w, h)=>{
       const resizedHeight = screenWidth * h / w; 
        this.setState(
          {
            width: screenWidth, 
            height: resizedHeight
          });
      });
    }

    receiveImage() {
      http.get('/result', {
        params: {
          idx: this.state.next_idx,
          'uuid': global.uuid,
          'item': global.curItem
        }
      })
      .then((response) => {
          console.log(response);

          if (response.data.status == 'Success'){
              // show image
              cur_idx = this.state.next_idx + 1;
              this.setState({
                next_idx: cur_idx,
                cur_url: response.data.url
              });
              this.resizeImage();
          }
          else {
            // text wait a while
            Alert.alert(
              'Next Image Not Ready',
              'please wait a while',
              [
                {text: 'OK', onPress: () => console.log('OK Pressed')},
              ],
              {cancelable: false},
            );
          }
        });
    }

    saveImage(){
      download(this.state.cur_url);
    }

    render() {
      return (
          <View style={styles.container}>
               <Image
                  style={{width: this.state.width, height: this.state.height}}
                  source={{uri: this.state.cur_url}}
              />
              <View style={styles.saveContainer}>
                <Button
                  onPress={()=>this.saveImage()}
                  title="Save"
                />
              </View>
              <View style={styles.nextContainer}>
                <Button
                  onPress={()=>this.receiveImage()}
                  title="Next"
                />
              </View>
          </View>
      );
    }
}

const styles = StyleSheet.create({
  container: {
      flex: 1,
      backgroundColor:'#F5FCFF'
  },
  saveContainer: {
      position: 'absolute',
      left: screenWidth*0.2,
      bottom: screenHeight*0.2
  },
  nextContainer: {
      position: 'absolute',
      right: screenWidth*0.2,
      bottom: screenHeight*0.2
  },
});
