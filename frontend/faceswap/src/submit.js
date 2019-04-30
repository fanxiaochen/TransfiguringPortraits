
import React, {Component} from 'react'
import {
    View,
    Button,
    TextInput,
    Text,
    StyleSheet
} from 'react-native'
import ImagePicker from 'react-native-image-picker';
import {formImageBody, submitFormData, curItem} from './upload'

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
        }
    }
    
    _onTakePhoto() {
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

    _onSubmit(){
        console.log(this.state)

     //   http.post('/item',{
     //       'item': this.state.item
     //   });
        
        curItem = this.state.item;
        submitFormData(this.state.avatarSource.uri, this.state.item)

        this.props.navigation.navigate('Wait')
    }


    render() {
        return (
            <View style={styles.container}>
                <TextInput
                    style={styles.input}
                    placeholder="Type a style for swapping, like obama"
                    onChangeText={(item) => this.setState({item})}
                />
                <Text style={{padding: 10, fontSize: 38}}>
                    {this.state.item.split(' ').map((word) => word && '🍕').join(' ')}
                </Text>
                <Button
                    style={styles.button}
                    onPress={()=>this._onTakePhoto()}
                    title='Take a selfie'
                />
                <Button
                    style={styles.button}
                    onPress={()=>this._onSubmit()}
                    title="submit"
                />
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
  input: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
});