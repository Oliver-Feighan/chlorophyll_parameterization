%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_301_chromophore_2 TDDFT with wB97XD functional

0 1
Mg   2.430   0.485   44.268
C   5.599   2.091   43.975
C   0.857   3.142   43.003
C   -0.170   -1.504   43.798
C   4.358   -2.393   45.336
N   3.173   2.332   43.552
C   4.410   2.846   43.556
C   4.481   4.327   43.171
C   2.959   4.740   43.001
C   2.272   3.331   43.183
C   2.370   5.838   43.874
C   5.357   4.460   41.832
C   5.183   3.321   40.804
C   5.529   3.597   39.314
O   6.093   4.563   38.838
O   5.116   2.581   38.580
N   0.608   0.773   43.435
C   0.070   1.980   43.055
C   -1.407   1.887   42.968
C   -1.706   0.530   43.174
C   -0.456   -0.155   43.399
C   -2.267   2.964   42.516
C   -3.081   -0.006   42.906
O   -3.938   0.843   42.556
C   -3.358   -1.528   43.015
N   2.202   -1.578   44.412
C   0.950   -2.055   44.327
C   0.894   -3.551   44.886
C   2.322   -3.863   45.301
C   2.983   -2.520   45.128
C   -0.218   -3.918   45.968
C   2.959   -5.063   44.478
C   3.938   -5.968   45.273
N   4.593   -0.126   44.560
C   5.109   -1.259   44.971
C   6.513   -1.070   45.380
C   6.720   0.237   44.932
C   5.540   0.807   44.449
C   7.292   -2.038   46.221
C   7.698   1.351   44.858
O   8.832   1.357   45.187
C   7.049   2.564   44.153
C   7.291   3.628   45.092
O   8.336   4.246   45.258
O   6.195   3.863   45.865
C   6.231   4.847   46.972
C   5.308   2.642   37.105
C   4.403   1.626   36.410
C   4.365   1.162   35.183
C   5.349   1.632   34.172
C   3.208   0.310   34.789
C   1.900   1.108   34.788
C   1.300   1.567   33.414
C   -0.189   2.033   33.505
C   -1.236   0.865   33.335
C   -0.462   3.218   32.569
C   -0.729   2.809   31.068
C   -2.176   3.197   30.654
C   -2.111   4.263   29.498
C   -2.085   5.761   30.071
C   -3.329   4.082   28.633
C   -2.962   4.123   27.110
C   -4.241   4.284   26.277
C   -4.254   3.430   25.039
C   -4.923   4.055   23.801
C   -4.577   1.901   25.171
H   0.211   3.971   42.706
H   -1.071   -2.120   43.787
H   4.901   -3.171   45.877
H   4.862   4.941   43.988
H   2.802   4.925   41.938
H   1.946   6.529   43.145
H   3.085   6.275   44.570
H   1.651   5.331   44.518
H   6.422   4.495   42.062
H   5.180   5.412   41.331
H   4.233   2.797   40.914
H   6.021   2.692   41.104
H   -2.354   2.935   41.429
H   -1.886   3.956   42.756
H   -3.221   2.968   43.042
H   -4.405   -1.813   42.907
H   -3.106   -1.620   44.071
H   -2.777   -2.166   42.348
H   0.617   -4.130   44.006
H   2.469   -4.030   46.368
H   -0.521   -2.948   46.361
H   0.234   -4.569   46.716
H   -1.143   -4.283   45.523
H   3.503   -4.587   43.662
H   2.083   -5.589   44.097
H   4.874   -5.904   44.717
H   3.562   -6.979   45.116
H   3.947   -5.716   46.333
H   6.832   -2.130   47.205
H   8.251   -1.545   46.383
H   7.330   -3.001   45.711
H   7.641   2.694   43.247
H   5.466   4.661   47.725
H   6.004   5.863   46.647
H   7.201   4.783   47.466
H   6.320   2.309   36.877
H   5.191   3.636   36.674
H   3.702   1.133   37.083
H   5.974   0.749   34.043
H   6.010   2.422   34.531
H   4.844   2.010   33.283
H   3.092   -0.601   35.377
H   3.336   0.155   33.717
H   2.080   2.104   35.191
H   1.139   0.705   35.457
H   1.590   0.783   32.714
H   1.901   2.390   33.027
H   -0.364   2.455   34.494
H   -2.059   1.183   32.695
H   -1.655   0.465   34.259
H   -0.685   0.044   32.878
H   0.409   3.872   32.615
H   -1.326   3.698   33.028
H   -0.706   1.731   30.907
H   0.061   3.296   30.497
H   -2.854   3.476   31.461
H   -2.638   2.286   30.274
H   -1.201   4.176   28.904
H   -2.306   5.691   31.136
H   -2.843   6.384   29.596
H   -1.133   6.244   29.853
H   -4.058   4.884   28.753
H   -3.888   3.178   28.876
H   -2.432   3.215   26.822
H   -2.373   5.027   26.955
H   -4.213   5.335   25.990
H   -5.210   4.111   26.744
H   -3.230   3.466   24.668
H   -5.173   5.085   24.056
H   -5.794   3.466   23.514
H   -4.263   4.032   22.935
H   -5.184   1.719   26.058
H   -3.589   1.443   25.126
H   -5.156   1.699   24.271

