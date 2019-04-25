import React, {Component} from 'react'
import {
    View,
    Text,
    StyleSheet
} from 'react-native'

import {http} from './upload'


export default class Swap extends Component {
    constructor(props){
        super(props);
        this.state = {
          idx: 0
        }

        this.receiveImage();

      }

    receiveImage() {
      http.get('/result', {
        params: {
          idx: this.state.idx
        }
      })
      .then((response) => {
          console.log(response);
        });
    }

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
