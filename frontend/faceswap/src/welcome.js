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
        <Text style={styles.welcome}>Face Swapping Demo</Text>
        <Button
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
    fontSize: 30,
    fontWeight: "bold",
    textAlign: 'center',
    margin: 30,
  },
});
