import React, {Component} from 'react'
import {
    View,
    Text,
    Button,
    StyleSheet
} from 'react-native'


export default class Welcome extends Component {

    render(){
        return (
      <View style={styles.container}>
        <Text style={styles.welcome}>A simple demo of face swapping</Text>
        <Button
            //onPress={()=>this.props.navigation.navigate('Photo')}
            onPress={()=>this.props.navigation.navigate('Submit')}
            title="Let's start"
        />
      </View>
        );
    }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  welcome: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
});
