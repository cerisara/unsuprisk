#!/bin/bash

mkdir test_reuters data logdir

cd ..
source myenv/bin/activate
cd src

nnn=0
for dp in 0.0; do
for corr in 0.0; do
for p0 in 0.2 0.3 0.4 0.1 0.49 0.01; do
for r in 3; do
for nc in 6; do
for repeat in 1; do
    echo "corr $corr nc $nc repeat $repeat r $r p0 $p0"
    echo "p0 = $p0" > datasets/globconfig.py
    echo 'lr = 0.0001' >> datasets/globconfig.py
    echo 'nep = 1000' >> datasets/globconfig.py
    echo "traincorr = $corr" >> datasets/globconfig.py
    echo 'devcorr = 0.5' >> datasets/globconfig.py
    echo "nheads = $r" >> datasets/globconfig.py
    echo "nc = $nc" >> datasets/globconfig.py
    echo "repeat = $repeat" >> datasets/globconfig.py
    echo "dropout = $dp" >> datasets/globconfig.py
    echo 'w2vcache = "data"' >> datasets/globconfig.py

    python main.py reuters cvdd_Net logdir data --device cpu --seed $repeat --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads $r --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class $nc  > tmplog.$nnn 2>&1
    cat tmplog.$nnn | grep 'TEST AUC CVDD' | tail -1 > tmploglst
    cvddauc=""`cat tmploglst | awk '{print $4}'`
    riskauc=""`cat tmploglst | awk '{print $6}'`
    echo "AUC r=$r nc=$nc CVDD=$cvddauc Eq4=$riskauc"

    nnn=$(($nnn + 1))
done
done
done
done
done
done


