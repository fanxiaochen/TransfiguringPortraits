import React, {Component} from 'react'
import {
    View,
    Button,
    StyleSheet
} from 'react-native'
import ImagePicker from 'react-native-image-picker';

const options = {
  title: 'Select Avatar',
  storageOptions: {
    skipBackup: true,
    path: 'images',
  },
};

export default class TakePhoto extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            avatarSource: {}
        }
    }
    
    _onPressButton() {
        ImagePicker.launchCamera(options, (response) => {
            if (response.didCancel) {
                console.log('User cancelled image picker');
            } else if (response.error) {
                console.log('ImagePicker Error: ', response.error);
            } else {
                const source = { uri: response.uri };
                console.log(source);
            
                // You can also display the image using data:
                // const source = { uri: 'data:image/jpeg;base64,' + response.data };
            
                this.setState({
                avatarSource: source,
                });

                this.props.navigation.navigate('Item');
            }
        });
    } 

    render() {
        return (
            <View style={styles.container}>
                <Button
                    onPress={()=>this._onPressButton()}
                    title='Take a selfie'
                />
            </View>
        )
    }
}

const styles = StyleSheet.create({
  container: {
   flex: 1,
   justifyContent: 'center',
  },
  buttonContainer: {
    margin: 20
  }});