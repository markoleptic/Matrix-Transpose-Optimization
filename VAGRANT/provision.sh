#!/bin/bash

apt-get update
apt-get upgrade -y

apt-get install -y build-essential

# I don't want to upset Dr. Veras too much so I am also installing emacs, but I put neovim first for a reason ;)
apt-get install -y neovim emacs git openssh-server

