#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1erPs4WlH9Hma5oU03BwsIkBa_DvZL73e' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1erPs4WlH9Hma5oU03BwsIkBa_DvZL73e" -O data.tar.gz && rm -rf /tmp/cookies.txt
tar -xf data.tar.gz
rm data.tar.gz