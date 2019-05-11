
import React, {Component} from 'react'
import {
    View,
    Button,
    TextInput,
    Text,
    Alert,
    Dimensions,
    StyleSheet
} from 'react-native'
import ImagePicker from 'react-native-image-picker';
import {submitFormData, uuidGen} from './upload'

let screenWidth = Dimensions.get('window').width;
let screenHeight = Dimensions.get('window').height;

const options = {
  title: 'Select Avatar',
  storageOptions: {
    skipBackup: true,
    path: 'images',
  },
};

export default class Submit extends Component {
    constructor(props){
        super(props);
        this.state = {
            avatarSource: {},
            item: '',
            uuid: ''
        }
    }
    
    onTakePhoto() {
        ImagePicker.launchCamera(options, (response) => {
            if (response.didCancel) {
                console.log('User cancelled image picker');
            } else if (response.error) {
                console.log('ImagePicker Error: ', response.error);
            } else {
                const source = { uri: response.uri };
            
                // You can also display the image using data:
                // const source = { uri: 'data:image/jpeg;base64,' + response.data };
            
                this.setState({
                avatarSource: source,
                });
                console.log(this.state);

               // formImageBody(response.uri);

               // this.props.navigation.navigate('Item');
            }
        });
    } 

    onSubmit(){
        console.log(this.state)

        if (this.state.item === ''){
            Alert.alert(
              'Error',
              'please type a style',
              [
                {text: 'OK', onPress: () => console.log('OK Pressed')},
              ],
              {cancelable: false},
            );
            return;
        }

        if (Object.entries(this.state.avatarSource).length === 0){
            Alert.alert(
              'Error',
              'please take a selfie',
              [
                {text: 'OK', onPress: () => console.log('OK Pressed')},
              ],
              {cancelable: false},
            );
            return;
        }


        if (this.state.uuid === ''){
            this.state.uuid = uuidGen.v1();
        }
        global.curItem = this.state.item;
        global.uuid = this.state.uuid;

        submitFormData(this.state.avatarSource.uri, this.state.item, this.state.uuid);

        this.props.navigation.navigate('Wait');
    }


    render() {
        return (
            <View style={styles.container}>
                <TextInput
                    style={styles.input}
                    autoFocus={true}
                    placeholder="Type a style for swapping, like obama"
                    onChangeText={(item) => this.setState({item})}
                />
                <Button
                    style={styles.photoButton}
                    onPress={()=>this.onTakePhoto()}
                    title='Take selfie'
                />
                <View style={styles.submitContainer}>
                    <Button
                        onPress={()=>this.onSubmit()}
                        color="#ff5c5c"
                        title="submit"
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
    position: 'absolute',
    top: 0,
    height: screenHeight / 2
  },
  input: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  photoButton: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  submitContainer: {
    justifyContent: 'center', 
    alignItems: 'center',
    position: 'absolute',
    bottom: 0 
  },
});