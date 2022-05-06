%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1751_chromophore_13 TDDFT with blyp functional

0 1
Mg   46.953   25.288   28.789
C   47.941   27.399   31.493
C   46.222   22.832   31.181
C   47.054   23.134   26.292
C   48.403   27.703   26.646
N   47.016   25.170   31.111
C   47.432   26.140   31.943
C   47.090   25.784   33.370
C   46.800   24.179   33.259
C   46.647   24.060   31.753
C   47.846   23.275   34.032
C   45.818   26.520   33.859
C   45.855   27.090   35.288
C   44.843   26.643   36.489
O   43.792   26.124   36.339
O   45.354   27.174   37.666
N   46.383   23.297   28.678
C   46.208   22.438   29.768
C   45.860   21.155   29.180
C   46.060   21.256   27.794
C   46.493   22.591   27.494
C   45.331   19.997   30.126
C   45.949   20.212   26.717
O   46.035   20.440   25.513
C   45.766   18.869   27.113
N   47.650   25.385   26.802
C   47.629   24.341   25.958
C   48.251   24.639   24.576
C   48.406   26.206   24.610
C   48.153   26.441   26.112
C   49.581   23.836   24.334
C   47.433   26.922   23.582
C   46.023   27.207   24.058
N   47.886   27.217   28.900
C   48.364   28.100   27.941
C   48.800   29.307   28.625
C   48.666   29.091   29.944
C   48.201   27.777   30.131
C   49.240   30.716   27.945
C   48.830   29.675   31.292
O   49.238   30.766   31.564
C   48.391   28.555   32.346
C   49.554   28.286   33.165
O   50.660   27.992   32.798
O   49.127   28.307   34.548
C   50.016   27.887   35.688
C   44.471   26.952   38.846
C   44.638   25.561   39.403
C   44.256   25.110   40.591
C   43.742   25.984   41.726
C   44.380   23.587   40.864
C   43.044   22.845   40.651
C   42.387   22.262   41.906
C   40.805   22.234   42.072
C   40.151   21.924   40.734
C   40.204   23.543   42.739
C   39.483   23.145   44.072
C   39.361   24.361   45.107
C   40.000   24.225   46.505
C   39.037   24.509   47.688
C   41.299   25.038   46.568
C   42.395   24.328   47.374
C   42.975   23.082   46.608
C   44.508   23.147   46.320
C   44.766   24.095   45.084
C   45.094   21.758   45.994
H   46.033   22.077   31.946
H   47.185   22.449   25.452
H   48.875   28.483   26.045
H   47.918   26.036   34.034
H   45.807   23.901   33.610
H   48.570   22.865   33.328
H   47.299   22.527   34.605
H   48.355   23.964   34.705
H   44.962   25.857   33.989
H   45.481   27.242   33.116
H   45.843   28.179   35.257
H   46.900   27.014   35.589
H   46.057   19.303   30.550
H   44.481   19.444   29.725
H   44.758   20.450   30.934
H   44.747   18.893   27.498
H   46.649   18.704   27.730
H   45.775   18.280   26.196
H   47.539   24.328   23.811
H   49.418   26.509   24.341
H   49.718   22.981   24.996
H   50.384   24.543   24.541
H   49.790   23.513   23.315
H   47.237   26.219   22.773
H   47.799   27.841   23.123
H   45.272   26.637   23.510
H   45.784   28.246   23.836
H   45.852   27.198   25.135
H   49.505   30.469   26.917
H   50.174   31.054   28.394
H   48.472   31.483   28.044
H   47.594   28.980   32.956
H   50.083   28.609   36.502
H   51.053   27.710   35.404
H   49.596   26.996   36.155
H   43.435   27.200   38.618
H   44.705   27.692   39.611
H   44.992   24.854   38.652
H   43.345   26.923   41.339
H   44.500   26.078   42.502
H   42.876   25.502   42.180
H   44.721   23.404   41.883
H   45.171   23.096   40.297
H   43.154   22.028   39.938
H   42.391   23.616   40.241
H   42.775   22.672   42.839
H   42.706   21.222   41.976
H   40.607   21.374   42.711
H   40.742   22.203   39.862
H   39.153   22.363   40.715
H   40.166   20.835   40.701
H   39.373   24.029   42.228
H   40.933   24.349   42.830
H   39.870   22.230   44.522
H   38.443   22.917   43.842
H   38.295   24.583   45.082
H   39.780   25.231   44.602
H   40.344   23.195   46.595
H   38.140   24.926   47.230
H   39.363   25.226   48.441
H   38.855   23.603   48.266
H   41.032   25.962   47.081
H   41.739   25.222   45.588
H   41.987   24.042   48.344
H   43.154   25.094   47.534
H   42.459   23.093   45.648
H   42.751   22.187   47.188
H   45.015   23.634   47.153
H   44.959   25.083   45.502
H   43.912   23.999   44.413
H   45.737   23.868   44.644
H   44.305   21.007   45.944
H   45.723   21.574   46.865
H   45.538   21.634   45.006

