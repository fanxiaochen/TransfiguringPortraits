import React, {Component} from 'react'
import {
    View,
    Text,
    StyleSheet
} from 'react-native'


export default class Swap extends Component {
    render(){
        return (
        <View>
            <Text style={styles.wait}>Show swapped results here</Text>
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
