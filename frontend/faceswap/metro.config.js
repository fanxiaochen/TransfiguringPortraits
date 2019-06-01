/**
 * Metro configuration for React Native
 * https://github.com/facebook/react-native
 *
 * @format
 */

const { resolve } = require('path');

module.exports = {
  watchFolders: [
    resolve(__dirname, '.'),        // your 'native' directory
    resolve(__dirname, '../../'),   // your '../' directory
  ],
  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: false,
      },
    }),
  },
};
