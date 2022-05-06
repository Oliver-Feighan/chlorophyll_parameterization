%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1651_chromophore_5 TDDFT with cam-b3lyp functional

0 1
Mg   24.238   -6.785   46.734
C   26.707   -4.578   45.829
C   22.120   -5.467   44.485
C   22.198   -9.588   47.130
C   26.816   -8.723   48.474
N   24.575   -5.393   45.083
C   25.660   -4.506   44.922
C   25.357   -3.529   43.843
C   23.849   -3.685   43.536
C   23.428   -4.923   44.426
C   22.976   -2.448   43.812
C   26.378   -3.625   42.704
C   26.765   -2.233   42.115
C   27.164   -2.407   40.596
O   28.284   -2.276   40.126
O   26.020   -2.771   39.854
N   22.352   -7.329   46.112
C   21.654   -6.579   45.167
C   20.375   -7.155   45.125
C   20.320   -8.379   45.867
C   21.649   -8.483   46.450
C   19.261   -6.499   44.312
C   19.152   -9.312   45.975
O   18.101   -8.990   45.358
C   19.095   -10.506   47.037
N   24.433   -8.849   47.691
C   23.431   -9.717   47.718
C   23.879   -10.961   48.590
C   25.200   -10.549   49.246
C   25.538   -9.336   48.391
C   22.849   -11.694   49.565
C   26.356   -11.648   49.405
C   26.775   -12.021   50.862
N   26.304   -6.698   47.129
C   27.172   -7.483   47.871
C   28.410   -6.763   48.054
C   28.211   -5.597   47.288
C   26.968   -5.618   46.752
C   29.589   -7.245   48.858
C   28.847   -4.369   46.882
O   29.927   -3.852   47.000
C   27.851   -3.597   46.024
C   27.479   -2.263   46.610
O   26.613   -2.039   47.466
O   28.391   -1.274   46.210
C   28.180   0.049   46.859
C   26.293   -3.061   38.412
C   25.846   -4.458   37.960
C   25.768   -4.871   36.630
C   26.115   -3.905   35.487
C   25.263   -6.236   36.375
C   23.857   -6.051   35.678
C   23.855   -6.392   34.176
C   22.925   -5.412   33.446
C   21.491   -5.909   33.529
C   23.339   -5.162   31.938
C   24.404   -4.037   31.795
C   25.749   -4.580   31.432
C   26.301   -4.135   30.037
C   27.716   -4.629   29.732
C   25.166   -4.300   28.915
C   24.789   -3.010   28.272
C   24.763   -2.993   26.695
C   23.342   -2.901   26.113
C   22.621   -4.256   26.285
C   23.328   -2.656   24.621
H   21.468   -4.880   43.835
H   21.551   -10.464   47.196
H   27.548   -9.180   49.143
H   25.399   -2.520   44.253
H   23.835   -3.984   42.488
H   23.643   -1.586   43.808
H   22.565   -2.416   44.820
H   22.275   -2.405   42.978
H   25.972   -4.165   41.849
H   27.256   -4.206   42.986
H   27.709   -2.022   42.617
H   25.990   -1.470   42.187
H   18.954   -7.170   43.510
H   19.567   -5.518   43.949
H   18.420   -6.331   44.986
H   19.468   -10.117   47.985
H   19.745   -11.264   46.600
H   18.112   -10.892   47.307
H   24.169   -11.678   47.822
H   25.085   -10.250   50.287
H   22.872   -12.708   49.165
H   21.901   -11.159   49.601
H   23.306   -11.731   50.554
H   27.271   -11.225   48.991
H   25.866   -12.515   48.962
H   27.743   -11.551   51.037
H   26.757   -13.109   50.915
H   26.075   -11.647   51.609
H   30.094   -8.148   48.515
H   29.192   -7.494   49.842
H   30.374   -6.498   48.977
H   28.265   -3.428   45.030
H   28.401   0.799   46.099
H   28.947   0.218   47.616
H   27.201   0.345   47.235
H   27.311   -2.904   38.055
H   25.708   -2.355   37.822
H   25.524   -5.082   38.794
H   26.899   -3.166   35.658
H   25.286   -3.199   35.540
H   26.269   -4.402   34.530
H   25.226   -6.823   37.294
H   26.035   -6.594   35.694
H   23.407   -5.071   35.838
H   23.154   -6.758   36.117
H   23.559   -7.440   34.143
H   24.787   -6.329   33.614
H   22.995   -4.388   33.814
H   21.403   -6.623   34.349
H   21.398   -6.466   32.597
H   20.731   -5.135   33.630
H   22.457   -4.942   31.336
H   23.726   -6.102   31.546
H   24.617   -3.504   32.722
H   23.980   -3.278   31.138
H   25.839   -5.624   31.731
H   26.467   -4.129   32.117
H   26.341   -3.059   30.206
H   28.329   -3.730   29.667
H   27.872   -5.056   28.741
H   28.032   -5.428   30.403
H   24.281   -4.683   29.424
H   25.448   -5.036   28.162
H   25.452   -2.224   28.636
H   23.798   -2.865   28.703
H   25.336   -3.869   26.394
H   25.233   -2.034   26.472
H   22.698   -2.222   26.672
H   23.108   -4.933   26.986
H   22.548   -4.836   25.365
H   21.659   -3.952   26.696
H   22.272   -2.585   24.360
H   23.777   -3.457   24.032
H   23.827   -1.691   24.535
