import React, {Component} from 'react'
import {
    View,
    Text,
    Button,
    Image,
    Alert,
    Dimensions,
    StyleSheet
} from 'react-native'

import {http} from './upload'

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
                //cur_url: 'https://reactnativecode.com/wp-content/uploads/2018/01/Error_Image_Android.png' 
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
      // TODO
    }

    render(){
        return (
        <View>
          <View style={styles.container}>
              <Image
                style={{width: this.state.width, height: this.state.height}}
                source={{uri: this.state.cur_url}}
              />
              <Button
                  style={styles.button}
                  onPress={()=>this.receiveImage()}
                  title='Next Image'
              />
          </View>
          <View style={styles.buttonContainer}>
              <Button
                  style={styles.saveButton}
                  onPress={()=>this.receiveImage()}
                  title='save'
              />
              <Button
                  style={styles.NextButton}
                  onPress={()=>this.receiveImage()}
                  title='next'
              />
          </View>
        </View>
        )
    }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  button: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    bottom: 0 
  },
  saveButton: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  NextButton: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
});
