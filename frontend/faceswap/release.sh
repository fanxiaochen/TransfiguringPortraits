#!/bin/bash

cd android
sudo ./gradlew clean && sudo ./gradlew assembleRelease
sudo cp app/build/outputs/apk/release/app-release.apk ../faceswap-v0.0.1.apk
cd ..
