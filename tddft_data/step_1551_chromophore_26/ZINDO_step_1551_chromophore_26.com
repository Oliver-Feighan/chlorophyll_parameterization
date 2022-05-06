%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1551_chromophore_26 ZINDO

0 1
Mg   -9.406   18.727   42.301
C   -5.872   18.253   42.391
C   -8.666   21.996   41.317
C   -12.611   19.233   41.729
C   -9.731   15.265   42.428
N   -7.499   19.917   41.910
C   -6.204   19.525   42.006
C   -5.227   20.652   41.748
C   -6.184   21.893   41.531
C   -7.533   21.206   41.518
C   -6.041   23.094   42.510
C   -4.118   20.400   40.643
C   -4.594   19.549   39.424
C   -4.069   19.796   38.070
O   -3.087   20.490   37.837
O   -4.903   19.237   37.133
N   -10.553   20.477   41.744
C   -10.082   21.752   41.379
C   -11.207   22.570   41.144
C   -12.415   21.814   41.296
C   -11.883   20.432   41.575
C   -11.012   24.073   40.886
C   -13.819   22.355   41.007
O   -14.070   23.481   40.595
C   -15.076   21.453   41.236
N   -10.876   17.366   41.642
C   -12.188   17.901   41.784
C   -13.178   16.707   41.993
C   -12.211   15.472   41.721
C   -10.838   16.042   41.965
C   -13.770   16.764   43.452
C   -12.300   14.945   40.240
C   -12.375   13.427   40.056
N   -8.091   17.003   42.412
C   -8.375   15.635   42.601
C   -7.066   14.974   42.896
C   -6.085   15.965   42.740
C   -6.760   17.185   42.503
C   -6.814   13.458   43.180
C   -4.676   16.174   42.761
O   -3.723   15.389   42.721
C   -4.472   17.703   42.564
C   -3.785   18.196   43.778
O   -2.631   18.641   43.781
O   -4.645   18.181   44.907
C   -4.000   18.647   46.111
C   -4.576   19.642   35.730
C   -5.588   18.941   34.793
C   -6.471   19.533   33.964
C   -6.653   21.001   33.693
C   -7.488   18.684   33.156
C   -8.997   18.960   33.437
C   -9.701   19.397   32.128
C   -11.053   20.233   32.499
C   -12.239   19.384   32.773
C   -11.421   21.284   31.410
C   -11.009   22.736   31.812
C   -9.925   23.233   30.834
C   -10.401   24.177   29.655
C   -9.452   25.318   29.274
C   -10.655   23.362   28.402
C   -11.639   24.004   27.447
C   -11.406   23.640   25.987
C   -11.666   24.824   25.036
C   -13.164   25.131   24.991
C   -11.079   24.645   23.598
H   -8.340   22.959   40.919
H   -13.702   19.235   41.696
H   -9.808   14.177   42.487
H   -4.749   20.765   42.721
H   -6.002   22.218   40.506
H   -7.013   23.352   42.931
H   -5.610   23.999   42.083
H   -5.420   22.804   43.357
H   -3.232   20.101   41.203
H   -3.853   21.399   40.297
H   -5.677   19.605   39.307
H   -4.445   18.497   39.666
H   -10.763   24.353   39.863
H   -10.295   24.494   41.591
H   -11.923   24.628   41.110
H   -15.013   21.053   42.248
H   -14.882   20.670   40.503
H   -16.012   21.958   40.995
H   -14.029   16.724   41.313
H   -12.484   14.722   42.463
H   -14.855   16.851   43.520
H   -13.452   17.668   43.969
H   -13.531   15.916   44.094
H   -11.429   15.358   39.732
H   -13.120   15.353   39.649
H   -11.502   13.115   39.483
H   -13.265   13.232   39.457
H   -12.350   12.831   40.969
H   -5.944   13.466   43.836
H   -6.565   13.028   42.210
H   -7.664   12.951   43.637
H   -3.853   17.895   41.687
H   -3.039   19.161   46.135
H   -3.947   17.844   46.846
H   -4.724   19.377   46.472
H   -3.557   19.342   35.486
H   -4.608   20.731   35.692
H   -5.651   17.860   34.916
H   -6.771   21.294   32.649
H   -5.820   21.559   34.119
H   -7.521   21.418   34.203
H   -7.435   17.624   33.406
H   -7.193   18.719   32.107
H   -9.161   19.661   34.256
H   -9.478   17.982   33.475
H   -10.009   18.485   31.617
H   -9.005   19.824   31.405
H   -10.842   20.752   33.434
H   -13.009   19.401   32.002
H   -12.802   19.855   33.579
H   -11.968   18.343   32.951
H   -12.475   21.205   31.143
H   -10.910   20.901   30.527
H   -10.680   22.870   32.842
H   -11.866   23.403   31.713
H   -9.414   22.351   30.446
H   -9.190   23.734   31.465
H   -11.367   24.556   29.990
H   -8.826   25.493   30.149
H   -10.046   26.231   29.226
H   -8.925   25.205   28.327
H   -10.952   22.324   28.553
H   -9.687   23.321   27.902
H   -11.694   25.079   27.617
H   -12.639   23.695   27.751
H   -12.209   22.907   25.905
H   -10.414   23.323   25.667
H   -11.176   25.686   25.488
H   -13.466   25.782   24.171
H   -13.501   25.761   25.814
H   -13.695   24.181   24.914
H   -10.466   23.746   23.539
H   -10.434   25.474   23.308
H   -11.865   24.505   22.856

