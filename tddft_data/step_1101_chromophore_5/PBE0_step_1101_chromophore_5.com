%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1101_chromophore_5 TDDFT with PBE1PBE functional

0 1
Mg   24.459   -7.578   46.115
C   26.635   -5.127   44.665
C   21.951   -6.274   44.245
C   22.763   -10.366   46.680
C   27.423   -9.024   47.454
N   24.386   -5.886   44.612
C   25.362   -5.017   44.224
C   24.826   -4.062   43.124
C   23.248   -4.341   43.276
C   23.130   -5.515   44.189
C   22.494   -3.088   43.845
C   25.463   -4.248   41.729
C   26.045   -3.018   40.959
C   25.578   -2.988   39.439
O   26.227   -2.577   38.496
O   24.235   -3.371   39.221
N   22.545   -8.193   45.678
C   21.662   -7.482   44.831
C   20.467   -8.260   44.770
C   20.619   -9.445   45.505
C   22.004   -9.358   45.965
C   19.276   -7.858   43.915
C   19.496   -10.474   45.779
O   18.338   -10.353   45.334
C   19.776   -11.649   46.678
N   24.969   -9.402   47.085
C   24.071   -10.437   47.083
C   24.735   -11.706   47.591
C   26.240   -11.308   47.908
C   26.287   -9.765   47.563
C   24.049   -12.349   48.842
C   27.337   -12.121   47.070
C   28.617   -12.446   47.817
N   26.573   -7.030   46.364
C   27.624   -7.777   46.829
C   28.849   -7.014   46.722
C   28.522   -5.915   45.933
C   27.154   -6.043   45.621
C   30.153   -7.450   47.277
C   28.982   -4.742   45.245
O   30.073   -4.142   45.192
C   27.755   -4.197   44.349
C   27.414   -2.713   44.547
O   27.415   -1.863   43.691
O   27.152   -2.461   45.882
C   26.633   -1.142   46.161
C   23.744   -3.329   37.858
C   23.704   -4.740   37.444
C   24.279   -5.239   36.312
C   25.292   -4.603   35.386
C   24.072   -6.691   36.099
C   22.747   -7.229   35.381
C   22.808   -7.208   33.846
C   21.743   -6.298   33.204
C   20.282   -6.974   33.131
C   22.186   -5.852   31.812
C   23.186   -4.671   31.815
C   24.613   -5.065   31.251
C   25.018   -4.162   30.065
C   26.251   -3.252   30.199
C   25.019   -4.952   28.755
C   25.089   -4.179   27.392
C   24.185   -4.831   26.201
C   23.242   -3.749   25.681
C   21.881   -4.245   26.047
C   23.260   -3.676   24.141
H   21.153   -5.785   43.683
H   22.305   -11.356   46.739
H   28.277   -9.546   47.890
H   25.060   -3.050   43.453
H   22.783   -4.585   42.321
H   21.770   -3.504   44.546
H   21.913   -2.676   43.020
H   23.155   -2.357   44.309
H   24.798   -4.878   41.138
H   26.305   -4.914   41.919
H   27.126   -3.031   41.096
H   25.715   -2.095   41.435
H   18.394   -8.069   44.520
H   19.238   -8.557   43.080
H   19.385   -6.847   43.524
H   20.373   -11.377   47.548
H   20.355   -12.425   46.178
H   18.795   -11.996   47.001
H   24.702   -12.396   46.748
H   26.537   -11.463   48.945
H   22.986   -12.112   48.888
H   24.436   -11.881   49.748
H   24.348   -13.394   48.768
H   27.684   -11.543   46.213
H   26.882   -13.005   46.622
H   29.387   -11.796   47.401
H   28.970   -13.465   47.662
H   28.367   -12.235   48.857
H   30.673   -8.186   46.663
H   29.957   -7.860   48.268
H   30.812   -6.590   47.402
H   27.895   -4.200   43.268
H   25.818   -0.874   45.488
H   27.388   -0.363   46.055
H   26.250   -1.220   47.178
H   24.287   -2.808   37.069
H   22.693   -3.059   37.963
H   22.988   -5.342   38.004
H   26.284   -4.789   35.798
H   25.287   -3.521   35.251
H   25.245   -5.005   34.374
H   24.042   -7.308   36.997
H   24.883   -7.163   35.544
H   21.945   -6.563   35.699
H   22.450   -8.232   35.690
H   22.572   -8.203   33.470
H   23.810   -6.904   33.541
H   21.615   -5.372   33.765
H   19.597   -6.315   33.665
H   20.233   -7.960   33.593
H   19.972   -7.042   32.088
H   21.223   -5.621   31.356
H   22.492   -6.775   31.318
H   23.369   -4.212   32.786
H   22.708   -3.900   31.210
H   24.604   -6.130   31.019
H   25.321   -4.791   32.034
H   24.280   -3.386   29.862
H   26.133   -2.384   29.550
H   27.167   -3.801   29.985
H   26.310   -2.865   31.216
H   24.306   -5.776   28.768
H   25.986   -5.443   28.867
H   26.091   -4.354   26.998
H   24.767   -3.138   27.408
H   23.686   -5.714   26.600
H   24.881   -5.294   25.501
H   23.467   -2.776   26.119
H   21.183   -3.409   25.996
H   21.785   -4.515   27.099
H   21.482   -5.033   25.408
H   23.534   -2.626   24.036
H   22.470   -4.031   23.479
H   24.124   -4.228   23.771

