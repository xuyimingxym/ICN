# ICN (Interactive Convolutional Network)
This is the sample code for TRB paper "ICN: Interactive Convolutional Network for Forecasting Travel Demand of Shared Micromobility".

NOTE: Sample code only. Not completely runnable. 

Run the code: 
```
python run_micromobility.py --dataset Chicago \
--horizon 1 --window_size 24 --batch_size 32 --hidden_size 0.5 --dropout 0.5 --lr 0.001 \
--model_name Chicago_Weather --tune True --epoch 80 --weather True
```
