import React, {Component} from 'react'
import {
    View,
    Text,
    StyleSheet
} from 'react-native'


export default class Waiting extends Component {
    render(){
        return (
        <View>
            <Text style={styles.wait}>Please waiting for a while...</Text>
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
  wait: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
});
