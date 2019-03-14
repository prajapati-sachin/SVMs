export PYTHONPATH="/home/sachin/libsvm-3.23/python:${PYTHONPATH}"
#!/bin/bash
a=1
b=2
# c=3
# d=4
aa=a
bb=b
cc=c


if [ $1 == $a ]; then 
    python3 q1.py $2 $3 $4 $5 
elif [ $1 == $b ]; then 
    if [ $4 == $aa ]; then 
        python3 svm.py $2 $3 $4 
    elif [ $4 == $bb ]; then 
        python3 svm.py $2 $3 $4 
    elif [ $4 == $cc ]; then 
        python3 libsvm.py $2 $3 $4     
    fi
    # python3 q2.py $2 $3 $4 
fi


