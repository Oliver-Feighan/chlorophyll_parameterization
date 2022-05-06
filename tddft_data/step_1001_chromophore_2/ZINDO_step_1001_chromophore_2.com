%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1001_chromophore_2 ZINDO

0 1
Mg   3.350   0.760   44.079
C   6.354   2.589   43.975
C   1.872   3.546   42.545
C   0.731   -1.064   43.474
C   5.283   -2.040   44.941
N   4.107   2.706   43.094
C   5.319   3.234   43.314
C   5.421   4.622   42.788
C   3.939   4.948   42.335
C   3.225   3.711   42.696
C   3.328   6.236   42.896
C   6.518   4.806   41.713
C   6.854   3.670   40.738
C   7.158   4.111   39.267
O   7.874   5.116   38.978
O   6.513   3.365   38.314
N   1.463   1.204   43.219
C   1.057   2.423   42.762
C   -0.288   2.288   42.375
C   -0.654   0.910   42.537
C   0.512   0.259   43.035
C   -1.059   3.379   41.734
C   -2.023   0.402   42.239
O   -2.874   1.144   41.843
C   -2.387   -1.067   42.329
N   3.105   -1.376   43.923
C   1.870   -1.835   43.768
C   1.798   -3.345   44.123
C   3.291   -3.683   44.574
C   3.937   -2.258   44.564
C   0.652   -3.896   45.186
C   4.201   -4.668   43.771
C   4.836   -5.956   44.369
N   5.364   0.302   44.528
C   5.960   -0.806   44.967
C   7.300   -0.533   45.362
C   7.513   0.820   45.030
C   6.293   1.284   44.514
C   8.332   -1.503   45.894
C   8.504   1.898   45.033
O   9.684   1.992   45.398
C   7.748   3.149   44.318
C   7.716   4.349   45.227
O   8.582   5.191   45.199
O   6.610   4.387   46.110
C   6.564   5.516   46.989
C   6.769   3.700   36.895
C   5.708   3.003   36.019
C   5.746   2.028   35.071
C   6.992   1.552   34.460
C   4.463   1.292   34.670
C   3.208   2.159   34.212
C   3.183   2.421   32.721
C   2.111   1.630   31.928
C   2.667   0.349   31.251
C   1.415   2.552   30.976
C   0.303   3.462   31.617
C   -1.110   3.113   31.071
C   -1.647   4.320   30.170
C   -2.374   5.386   30.954
C   -2.311   3.789   28.922
C   -1.406   3.340   27.812
C   -1.854   3.797   26.429
C   -2.747   2.720   25.643
C   -2.408   2.833   24.126
C   -4.200   2.981   25.972
H   1.437   4.415   42.047
H   -0.217   -1.601   43.554
H   5.662   -2.941   45.429
H   5.589   5.408   43.524
H   3.835   5.006   41.252
H   4.140   6.852   43.282
H   2.566   6.002   43.639
H   2.897   6.773   42.051
H   7.383   5.111   42.300
H   6.173   5.653   41.120
H   5.998   3.026   40.537
H   7.843   3.281   40.980
H   -1.995   3.452   42.288
H   -1.138   3.282   40.652
H   -0.705   4.404   41.840
H   -2.534   -1.192   43.402
H   -1.561   -1.634   41.898
H   -3.251   -1.342   41.724
H   1.664   -3.765   43.126
H   3.220   -4.185   45.539
H   0.385   -2.950   45.657
H   0.961   -4.708   45.845
H   -0.190   -4.358   44.669
H   5.030   -4.115   43.329
H   3.540   -5.124   43.035
H   4.725   -6.802   43.690
H   4.481   -6.296   45.342
H   5.914   -5.852   44.487
H   8.218   -1.488   46.978
H   9.295   -1.068   45.627
H   8.199   -2.495   45.460
H   8.412   3.298   43.466
H   7.489   6.077   47.123
H   6.459   4.979   47.932
H   5.711   6.156   46.767
H   7.794   3.368   36.734
H   6.724   4.741   36.576
H   4.697   3.106   36.414
H   6.946   0.471   34.597
H   7.806   2.061   34.975
H   6.905   1.954   33.451
H   4.052   0.617   35.420
H   4.751   0.712   33.793
H   3.406   3.131   34.663
H   2.205   1.855   34.509
H   4.176   2.276   32.297
H   3.034   3.498   32.641
H   1.421   1.348   32.724
H   3.656   0.076   31.619
H   2.624   0.402   30.163
H   2.069   -0.487   31.614
H   0.991   2.017   30.126
H   2.167   3.236   30.582
H   0.548   4.523   31.567
H   0.375   3.315   32.695
H   -1.749   3.089   31.954
H   -1.187   2.173   30.525
H   -0.855   4.925   29.729
H   -2.859   6.059   30.248
H   -1.695   6.002   31.544
H   -3.130   4.937   31.598
H   -3.065   4.556   28.745
H   -2.945   2.939   29.176
H   -1.352   2.252   27.773
H   -0.378   3.620   28.041
H   -0.900   3.980   25.933
H   -2.369   4.753   26.511
H   -2.494   1.777   26.129
H   -1.545   3.485   23.995
H   -3.152   3.147   23.394
H   -2.269   1.801   23.804
H   -4.696   2.017   26.089
H   -4.779   3.556   25.249
H   -4.342   3.538   26.898

