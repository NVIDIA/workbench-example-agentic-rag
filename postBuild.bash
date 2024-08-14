#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it
sudo -E apt-get update && sudo -E apt-get install ffmpeg libsm6 libxext6 -y
