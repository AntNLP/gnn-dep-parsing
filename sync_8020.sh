#!/bin/zsh

rsync -avz -e 'ssh -p 8020' gnn-dep-parser/ taoji@139.196.228.41:/home/taoji/data/Github/gnn-dep-parsing
