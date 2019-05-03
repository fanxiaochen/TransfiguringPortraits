import React, {Component} from 'react'
import {
    View,
    Text,
    StyleSheet
} from 'react-native'

import {http, uuid, curItem} from './upload'


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
              this.props.navigation.navigate('Swap')
          }
        });

    }

 //   waitingForImage() {
 //     http.get('/result', {
 //       params: {
 //         img_idx: 0
 //       }
 //     })
 //     .then((response) => {
 //         //check if status is completed, if it is stop polling 
 //         if(response.data.status = 'completed') {
 //               clearInterval(this.pollInterval) //won't be polled anymore 
 //         }
 //         this.status = response; 
 //       });
 //   }

    mounted() {
      if(this.state.status != 'Accepted') {
        this.pollInterval = setInterval(()=>{this.hasSwapped()}, 2000) //save reference to the interval
        setTimeout(() => {clearInterval(this.pollInterval)}, 600000) //stop polling after an hour
      }
    }

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
