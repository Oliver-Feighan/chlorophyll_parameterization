%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_901_chromophore_24 TDDFT with blyp functional

0 1
Mg   -0.279   44.056   24.992
C   1.742   43.810   27.738
C   -3.036   43.264   27.020
C   -2.302   43.949   22.244
C   2.491   44.468   22.949
N   -0.583   43.630   27.145
C   0.389   43.609   28.147
C   -0.222   43.348   29.505
C   -1.728   42.963   29.138
C   -1.742   43.339   27.665
C   -2.230   41.481   29.472
C   0.056   44.472   30.525
C   0.365   44.015   31.980
C   -0.184   42.726   32.468
O   0.383   41.609   32.389
O   -1.467   42.914   32.899
N   -2.379   43.767   24.692
C   -3.345   43.464   25.641
C   -4.691   43.473   24.981
C   -4.488   43.744   23.573
C   -2.984   43.841   23.432
C   -5.961   43.232   25.803
C   -5.505   43.953   22.548
O   -5.239   44.190   21.377
C   -7.042   43.887   22.927
N   0.025   44.350   22.862
C   -0.948   44.063   21.978
C   -0.370   43.945   20.593
C   1.124   44.426   20.782
C   1.291   44.483   22.295
C   -0.519   42.430   20.171
C   1.415   45.758   20.044
C   0.952   46.995   20.885
N   1.668   44.084   25.210
C   2.681   44.272   24.305
C   3.970   44.102   24.998
C   3.632   43.911   26.322
C   2.205   43.988   26.398
C   5.315   44.042   24.365
C   4.231   43.716   27.593
O   5.389   43.545   28.024
C   3.017   43.716   28.542
C   3.149   42.519   29.374
O   2.829   41.367   29.070
O   3.575   42.893   30.630
C   3.702   41.975   31.749
C   -2.082   41.856   33.642
C   -3.522   42.220   33.942
C   -4.247   41.666   34.893
C   -3.888   40.365   35.535
C   -5.641   42.026   35.317
C   -5.779   42.569   36.745
C   -5.614   44.153   36.789
C   -6.580   44.863   37.704
C   -5.985   46.023   38.477
C   -7.889   45.162   36.982
C   -9.222   44.492   37.484
C   -9.862   43.523   36.457
C   -10.394   42.184   36.998
C   -10.652   41.052   35.998
C   -11.726   42.462   37.794
C   -12.078   41.401   38.826
C   -13.457   40.702   38.550
C   -13.387   39.275   39.055
C   -12.498   38.275   38.297
C   -14.844   38.788   39.340
H   -3.807   42.978   27.739
H   -2.915   43.904   21.342
H   3.330   44.557   22.256
H   0.311   42.475   29.882
H   -2.461   43.533   29.710
H   -2.736   41.098   28.586
H   -2.820   41.403   30.386
H   -1.404   40.853   29.803
H   -0.829   45.108   30.566
H   0.879   45.111   30.205
H   0.059   44.885   32.561
H   1.454   44.042   32.012
H   -6.362   42.276   25.468
H   -6.602   44.100   25.655
H   -5.775   43.213   26.877
H   -7.278   44.723   23.586
H   -7.351   42.966   23.420
H   -7.679   44.028   22.054
H   -0.867   44.564   19.846
H   1.840   43.759   20.302
H   0.468   41.976   20.078
H   -0.949   42.232   19.189
H   -1.165   41.830   20.811
H   0.871   45.747   19.100
H   2.477   45.818   19.804
H   0.377   46.726   21.771
H   0.301   47.600   20.253
H   1.711   47.749   21.092
H   5.279   43.510   23.415
H   6.014   43.526   25.022
H   5.729   45.049   24.314
H   3.118   44.610   29.157
H   4.722   41.929   32.129
H   3.236   41.037   31.444
H   3.156   42.387   32.598
H   -1.544   41.582   34.549
H   -2.142   40.952   33.037
H   -3.877   43.137   33.471
H   -3.294   40.388   36.448
H   -3.189   39.833   34.889
H   -4.776   39.760   35.719
H   -6.259   41.136   35.207
H   -5.835   42.827   34.603
H   -5.060   42.052   37.381
H   -6.748   42.174   37.048
H   -5.353   44.664   35.862
H   -4.645   44.134   37.286
H   -6.889   44.204   38.515
H   -4.961   46.222   38.160
H   -6.000   45.955   39.564
H   -6.498   46.919   38.127
H   -7.703   44.825   35.962
H   -8.048   46.240   36.997
H   -9.976   45.261   37.649
H   -8.997   43.910   38.378
H   -9.230   43.274   35.605
H   -10.681   44.021   35.936
H   -9.602   41.787   37.632
H   -11.712   40.821   35.899
H   -10.041   40.195   36.282
H   -10.321   41.361   35.007
H   -12.456   42.677   37.014
H   -11.694   43.446   38.262
H   -12.320   41.928   39.749
H   -11.245   40.720   39.001
H   -13.664   40.731   37.480
H   -14.140   41.356   39.090
H   -12.818   39.266   39.984
H   -11.665   37.855   38.860
H   -12.189   38.894   37.455
H   -12.999   37.448   37.794
H   -15.525   39.252   38.627
H   -15.057   39.081   40.368
H   -14.741   37.713   39.188

