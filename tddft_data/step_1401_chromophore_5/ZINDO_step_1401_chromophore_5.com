%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1401_chromophore_5 ZINDO

0 1
Mg   23.900   -6.646   46.068
C   26.279   -4.293   44.855
C   21.565   -5.051   44.116
C   21.989   -9.251   46.627
C   26.534   -8.121   47.815
N   23.996   -4.870   44.588
C   25.068   -4.147   44.248
C   24.701   -3.297   42.999
C   23.138   -3.311   43.037
C   22.863   -4.453   43.996
C   22.484   -1.987   43.497
C   25.318   -3.910   41.690
C   25.876   -3.006   40.589
C   25.251   -3.048   39.195
O   24.425   -2.287   38.731
O   25.789   -4.180   38.578
N   21.971   -7.115   45.430
C   21.125   -6.317   44.683
C   19.791   -6.873   44.674
C   19.891   -8.071   45.459
C   21.311   -8.181   45.877
C   18.606   -6.317   44.123
C   18.746   -8.999   45.686
O   17.698   -8.853   45.087
C   18.883   -10.278   46.581
N   24.213   -8.380   47.149
C   23.238   -9.336   47.304
C   23.677   -10.507   48.122
C   25.138   -10.119   48.572
C   25.303   -8.718   47.894
C   22.644   -10.952   49.278
C   26.326   -11.074   48.178
C   27.338   -11.476   49.261
N   26.009   -6.346   46.340
C   26.964   -6.961   47.107
C   28.273   -6.295   46.994
C   28.067   -5.278   46.026
C   26.650   -5.317   45.754
C   29.482   -6.769   47.650
C   28.717   -4.204   45.335
O   29.864   -3.743   45.282
C   27.504   -3.417   44.609
C   27.356   -2.055   45.221
O   26.696   -1.164   44.786
O   28.009   -1.985   46.409
C   27.962   -0.666   47.016
C   25.297   -4.448   37.255
C   26.123   -5.519   36.621
C   26.364   -5.790   35.355
C   25.544   -5.221   34.256
C   27.526   -6.733   34.867
C   27.199   -8.240   34.978
C   27.358   -9.064   33.654
C   26.153   -8.928   32.702
C   25.098   -9.994   32.999
C   26.530   -8.925   31.209
C   26.062   -7.627   30.450
C   27.133   -6.555   30.169
C   26.429   -5.340   29.536
C   27.038   -3.994   29.844
C   26.594   -5.516   28.001
C   25.237   -5.696   27.269
C   24.765   -4.401   26.505
C   23.194   -4.292   26.397
C   22.681   -3.364   25.299
C   22.741   -3.807   27.833
H   20.831   -4.483   43.541
H   21.335   -10.109   46.792
H   27.306   -8.726   48.295
H   25.136   -2.298   43.023
H   22.650   -3.558   42.094
H   22.155   -2.019   44.536
H   21.574   -1.963   42.898
H   23.122   -1.131   43.279
H   24.523   -4.501   41.236
H   26.148   -4.560   41.967
H   26.949   -3.187   40.526
H   25.773   -1.965   40.897
H   18.481   -6.882   43.199
H   18.676   -5.230   44.088
H   17.836   -6.589   44.846
H   19.266   -9.972   47.554
H   19.593   -10.872   46.005
H   17.986   -10.875   46.742
H   23.730   -11.335   47.415
H   25.158   -10.067   49.661
H   21.777   -10.295   49.357
H   23.021   -11.107   50.289
H   22.159   -11.872   48.952
H   26.838   -10.704   47.290
H   25.869   -12.038   47.958
H   27.631   -12.514   49.104
H   26.869   -11.421   50.243
H   28.196   -10.806   49.209
H   30.298   -6.737   46.928
H   29.221   -7.793   47.916
H   29.784   -6.245   48.557
H   27.775   -3.140   43.590
H   26.968   -0.270   47.221
H   28.564   -0.060   46.339
H   28.469   -0.821   47.968
H   25.347   -3.563   36.621
H   24.267   -4.792   37.342
H   26.680   -6.216   37.248
H   26.119   -4.544   33.624
H   24.652   -4.857   34.765
H   25.335   -6.064   33.597
H   28.423   -6.627   35.478
H   27.778   -6.397   33.860
H   26.162   -8.319   35.307
H   27.899   -8.701   35.675
H   27.401   -10.103   33.980
H   28.317   -8.769   33.229
H   25.820   -7.905   32.877
H   24.630   -10.486   32.146
H   24.204   -9.500   33.381
H   25.369   -10.762   33.723
H   26.072   -9.740   30.648
H   27.615   -8.950   31.115
H   25.375   -7.115   31.123
H   25.544   -7.957   29.549
H   27.981   -6.984   29.635
H   27.419   -6.080   31.107
H   25.407   -5.151   29.865
H   27.225   -4.028   30.918
H   26.265   -3.263   29.608
H   28.011   -3.775   29.404
H   27.223   -6.392   27.845
H   27.091   -4.639   27.586
H   24.489   -6.197   27.884
H   25.427   -6.451   26.507
H   25.205   -4.532   25.517
H   25.209   -3.566   27.048
H   22.857   -5.284   26.096
H   22.581   -2.375   25.748
H   21.809   -3.754   24.774
H   23.469   -3.214   24.561
H   22.140   -4.523   28.395
H   22.065   -2.984   27.604
H   23.617   -3.427   28.358

