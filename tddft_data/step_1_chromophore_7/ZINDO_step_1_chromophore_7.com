%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1_chromophore_7 ZINDO

0 1
Mg   26.070   0.323   29.175
C   27.930   0.003   32.172
C   23.237   0.373   31.222
C   24.200   0.735   26.508
C   28.899   -0.559   27.356
N   25.688   0.233   31.426
C   26.587   0.253   32.449
C   25.993   0.251   33.868
C   24.435   0.248   33.454
C   24.487   0.270   31.955
C   23.634   -1.023   33.943
C   26.399   1.509   34.703
C   27.306   1.478   35.927
C   27.075   2.414   37.106
O   26.799   3.589   37.103
O   27.375   1.786   38.263
N   24.004   0.606   28.874
C   22.986   0.524   29.835
C   21.724   0.613   29.135
C   21.973   0.880   27.791
C   23.443   0.804   27.683
C   20.468   0.648   29.937
C   21.028   1.082   26.607
O   21.392   1.281   25.444
C   19.585   1.184   26.910
N   26.429   0.025   27.200
C   25.570   0.462   26.230
C   26.301   0.604   24.852
C   27.706   0.101   25.157
C   27.680   -0.244   26.672
C   25.559   -0.108   23.674
C   28.733   1.183   24.790
C   29.963   0.773   24.065
N   28.005   -0.282   29.610
C   29.056   -0.580   28.780
C   30.245   -0.868   29.525
C   29.924   -0.560   30.937
C   28.520   -0.198   30.898
C   31.586   -1.337   28.935
C   30.407   -0.487   32.339
O   31.558   -0.586   32.797
C   29.115   -0.111   33.175
C   29.058   -1.284   34.103
O   28.531   -2.375   33.922
O   29.560   -0.978   35.268
C   29.754   -2.038   36.287
C   27.261   2.534   39.514
C   26.102   2.141   40.365
C   26.090   1.345   41.476
C   27.315   0.874   42.171
C   24.847   1.235   42.234
C   24.605   2.210   43.362
C   23.913   1.429   44.521
C   23.963   2.248   45.945
C   24.117   1.463   47.221
C   22.850   3.290   45.917
C   22.815   4.266   47.141
C   22.011   5.568   46.931
C   22.651   6.867   47.491
C   21.565   7.621   48.307
C   23.252   7.825   46.290
C   22.147   8.566   45.514
C   22.397   10.102   45.528
C   21.929   10.612   44.095
C   22.999   10.603   42.980
C   21.237   12.047   44.187
H   22.467   0.529   31.980
H   23.696   0.816   25.543
H   29.793   -0.764   26.764
H   26.263   -0.696   34.337
H   23.827   1.112   33.724
H   24.218   -1.884   34.267
H   22.997   -1.390   33.139
H   22.991   -0.724   34.771
H   25.469   1.946   35.067
H   26.901   2.237   34.067
H   28.234   1.847   35.490
H   27.279   0.421   36.192
H   19.927   1.584   29.805
H   20.678   0.536   31.001
H   19.763   -0.138   29.668
H   19.433   1.902   27.716
H   19.189   0.275   27.363
H   18.974   1.535   26.079
H   26.253   1.681   24.692
H   27.966   -0.748   24.525
H   24.561   -0.467   23.925
H   26.086   -0.978   23.284
H   25.565   0.535   22.794
H   29.004   1.586   25.765
H   28.349   1.977   24.150
H   30.780   1.143   24.684
H   29.933   1.134   23.037
H   30.142   -0.302   24.035
H   31.600   -2.426   28.910
H   32.389   -0.931   29.550
H   31.717   -1.014   27.902
H   29.351   0.804   33.718
H   28.833   -2.533   36.593
H   30.271   -1.703   37.186
H   30.410   -2.865   36.017
H   27.069   3.601   39.394
H   28.241   2.485   39.989
H   25.156   2.650   40.183
H   27.396   1.361   43.143
H   28.214   1.069   41.586
H   27.271   -0.200   42.356
H   24.834   0.211   42.608
H   24.085   1.281   41.456
H   23.995   3.006   42.936
H   25.490   2.688   43.783
H   24.331   0.449   44.751
H   22.854   1.341   44.279
H   24.879   2.837   45.912
H   24.858   1.827   47.934
H   24.256   0.397   47.044
H   23.099   1.595   47.588
H   21.936   2.721   46.086
H   22.810   3.756   44.932
H   23.813   4.525   47.493
H   22.402   3.710   47.983
H   21.067   5.387   47.445
H   21.804   5.709   45.870
H   23.521   6.634   48.104
H   21.837   8.677   48.304
H   21.567   7.348   49.362
H   20.538   7.443   47.987
H   23.751   7.184   45.563
H   23.951   8.476   46.815
H   21.138   8.454   45.910
H   22.077   8.186   44.495
H   23.455   10.185   45.776
H   21.848   10.524   46.371
H   21.096   10.017   43.718
H   24.032   10.427   43.281
H   23.099   11.552   42.455
H   22.762   9.823   42.256
H   21.041   12.246   45.240
H   20.247   12.030   43.731
H   21.927   12.721   43.679

