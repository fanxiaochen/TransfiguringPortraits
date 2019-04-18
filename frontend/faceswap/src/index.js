import React from 'react'

import TakePhoto from './photo'
import Item from './item'
import Waiting from './waiting'
import Swap from './swap'

import {
    createStackNavigator, 
    createAppContainer
} from 'react-navigation'

const AppNavigator = createStackNavigator({
    Home: {
      screen: TakePhoto,
    },
    Item: {
      screen: Item,
    },
    Wait: {
        screen: Waiting,
    },
    Swap: {
        screen: Swap
    }
    }, {
        initialRouteName: 'Home',
    });
  
  export default createAppContainer(AppNavigator); 


