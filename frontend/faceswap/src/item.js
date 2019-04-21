
import React, {Component} from 'react'
import {
    View,
    TextInput,
    Text,
    Button,
    StyleSheet
} from 'react-native'


export default class Item extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            item: ''
        }
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
                onPress={()=>this.props.navigation.navigate('Wait')}
                title="submit"
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
  input: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
});