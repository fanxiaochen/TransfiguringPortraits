import React, {Component} from 'react'
import {
    View,
    Text,
    Alert,
    StyleSheet
} from 'react-native'

import {http} from './upload'


export default class Waiting extends Component {
    constructor(props){
        super(props);
        this.state = {
          status: 'WA'
        }

        this.mounted();
    }

    hasSwapped() {
      console.log('try get');
      http.get('/swapped', {
        params: {
          'uuid': global.uuid,
          'item': global.curItem 
        }
      })
      .then((response) => {
          this.setState({
            status: response.data.status,
          });
          console.log(this.state);

          //check if status is completed, if it is stop polling 
          if(response.data.status === 'Accepted') {
              clearInterval(this.pollInterval) //won't be polled anymore 
              clearTimeout(this.timeout)
              this.props.navigation.navigate('Swap')
          }
        });

    }

    mounted() {
      if(this.state.status != 'Accepted') {
        this.pollInterval = setInterval(()=>{this.hasSwapped()}, 2000) //save reference to the interval
        this.timeout = setTimeout(() => {
          clearInterval(this.pollInterval);
          Alert.alert(
            'Timeout',
            'could not connect to server',
            [
              {text: 'OK', onPress: () => console.log('OK Pressed')},
            ],
            {cancelable: false},
          );
        }, 60000) //stop polling after an min
      }
    }

    render(){
        return (
        <View style={styles.container}>
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
