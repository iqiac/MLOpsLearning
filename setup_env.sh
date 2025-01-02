#!/bin/bash

USER_NAME=$(whoami)
USER_UID=$(id -u $USER_NAME)
USER_GID=$(id -g $USER_NAME)
COUNT=0
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  COUNT=all
fi

echo "USER_NAME=$USER_NAME
USER_UID=$USER_UID
USER_GID=$USER_GID
COUNT=$COUNT" > .env

exit 0