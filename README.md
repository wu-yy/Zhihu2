# Zhihu2
CCIR 2018 知乎预测

python main.py main --max_epoch=5 --plot_every=100 --env='inception-word' --weight=1 --model='CNNText_inception'  --zhuge=False --num-workers=4 --batch-size=512 --model-path=None --lr2=0 --lr=1e-3 --lr-decay=0.8  --decay-every=2500 --title-dim=1200 --content-dim=1200 --type_='word' --augument=False

python main.py main --max_epoch=5 --plot_every=100 --env='lstm_char' --weight=1 --model='LSTMText'  --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5 --decay_every=10000  --type_='char'   --zhuge=False --linear-hidden-size=2000 --hidden-size=256 --kmax-pooling=3   --num-layers=3  --augument=False
