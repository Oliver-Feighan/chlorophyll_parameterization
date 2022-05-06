%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1801_chromophore_19 TDDFT with cam-b3lyp functional

0 1
Mg   24.858   49.895   27.189
C   22.951   51.221   29.894
C   27.367   49.385   29.427
C   26.951   49.307   24.629
C   22.424   51.101   24.977
N   25.076   50.253   29.345
C   24.124   50.639   30.340
C   24.641   50.619   31.774
C   26.142   50.048   31.515
C   26.231   49.984   29.977
C   27.373   50.780   32.162
C   23.823   49.848   32.855
C   24.473   49.625   34.240
C   23.882   50.223   35.616
O   22.956   51.003   35.772
O   24.519   49.599   36.675
N   26.903   49.345   27.099
C   27.646   48.978   28.140
C   28.847   48.347   27.564
C   28.802   48.332   26.148
C   27.512   49.046   25.870
C   29.903   47.827   28.477
C   29.786   47.934   25.135
O   29.719   48.225   23.980
C   30.955   47.091   25.574
N   24.689   50.161   25.113
C   25.688   49.874   24.267
C   25.464   50.442   22.897
C   23.952   50.858   22.918
C   23.637   50.800   24.408
C   26.465   51.638   22.687
C   23.041   49.999   21.967
C   22.467   48.631   22.576
N   23.019   50.945   27.346
C   22.152   51.342   26.362
C   21.044   52.039   26.976
C   21.272   51.961   28.333
C   22.484   51.325   28.540
C   19.914   52.661   26.212
C   20.779   52.344   29.598
O   19.697   52.893   29.890
C   21.892   51.989   30.616
C   22.260   53.303   31.129
O   22.647   54.243   30.530
O   22.025   53.307   32.535
C   22.183   54.623   33.253
C   23.897   49.888   37.949
C   24.826   49.432   39.092
C   26.038   49.952   39.418
C   26.775   51.021   38.679
C   26.676   49.654   40.749
C   27.208   48.193   40.769
C   26.390   47.095   41.527
C   27.233   46.242   42.517
C   27.517   44.955   41.728
C   26.646   45.833   43.867
C   27.488   46.144   45.052
C   26.576   46.861   46.153
C   26.984   48.272   46.497
C   26.210   48.733   47.683
C   27.087   49.322   45.301
C   28.543   49.589   44.885
C   29.192   50.862   45.645
C   29.477   52.072   44.681
C   28.227   52.866   44.169
C   30.378   51.682   43.474
H   28.149   49.058   30.116
H   27.475   48.962   23.736
H   21.715   51.448   24.222
H   24.802   51.634   32.137
H   26.113   49.037   31.923
H   27.992   49.949   32.501
H   26.952   51.356   32.987
H   27.928   51.346   31.414
H   23.466   48.933   32.382
H   22.900   50.415   32.981
H   25.482   50.029   34.161
H   24.399   48.541   34.330
H   30.039   46.810   28.110
H   29.669   47.871   29.541
H   30.926   48.178   28.339
H   30.644   46.117   25.952
H   31.526   47.696   26.279
H   31.648   46.905   24.754
H   25.574   49.605   22.208
H   23.866   51.878   22.543
H   27.508   51.389   22.880
H   26.069   52.532   23.168
H   26.355   51.771   21.610
H   23.695   49.815   21.114
H   22.186   50.636   21.739
H   22.935   48.377   23.526
H   22.562   47.840   21.832
H   21.398   48.757   22.752
H   19.819   52.415   25.155
H   20.050   53.739   26.300
H   18.938   52.448   26.648
H   21.445   51.373   31.396
H   22.922   55.176   32.674
H   22.586   54.491   34.257
H   21.299   55.257   33.307
H   22.946   49.378   38.104
H   23.727   50.961   38.030
H   24.226   48.732   39.673
H   27.857   50.926   38.764
H   26.411   51.945   39.129
H   26.552   50.931   37.616
H   26.013   49.852   41.591
H   27.559   50.282   40.867
H   28.161   48.252   41.294
H   27.320   47.821   39.750
H   25.948   46.449   40.768
H   25.538   47.412   42.129
H   28.226   46.674   42.646
H   28.144   44.264   42.291
H   28.057   45.257   40.830
H   26.661   44.352   41.426
H   26.318   44.793   43.869
H   25.699   46.361   43.980
H   28.285   46.833   44.773
H   27.970   45.247   45.439
H   26.528   46.248   47.053
H   25.516   46.976   45.924
H   27.933   48.074   46.995
H   25.668   49.597   47.298
H   26.903   49.020   48.475
H   25.404   48.088   48.032
H   26.637   50.282   45.556
H   26.635   48.872   44.417
H   28.523   49.739   43.806
H   29.078   48.668   45.118
H   30.148   50.542   46.059
H   28.621   51.243   46.492
H   30.067   52.757   45.290
H   27.315   52.319   44.408
H   28.284   53.250   43.151
H   28.168   53.808   44.713
H   29.958   52.059   42.541
H   30.385   50.593   43.443
H   31.393   52.038   43.651
