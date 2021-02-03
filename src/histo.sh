#!/bin/bash

d=0.5
if [ $# -ge 2 ]; then
  echo "2parms $1 $2"
  d=$2
fi

rm -f /tmp/pp.gp
touch /tmp/pp.gp
echo 'set terminal x11 persist'>>/tmp/pp.gp
echo 'binwidth='$d>>/tmp/pp.gp
echo 'set boxwidth binwidth'>>/tmp/pp.gp
echo 'bin(x,width)=width*floor(x/width)+ binwidth/2.0'>>/tmp/pp.gp
echo 'set terminal jpeg'>>/tmp/pp.gp
echo 'set output "h.jpg"'>>/tmp/pp.gp
echo 'plot [0:1][0:600] "'$1'" using (bin($1,binwidth)):(1.0) smooth freq with boxes fs solid 0.25 notitle'>>/tmp/pp.gp
gnuplot /tmp/pp.gp

