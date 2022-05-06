%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_601_chromophore_20 TDDFT with PBE1PBE functional

0 1
Mg   7.343   57.679   41.586
C   6.444   54.292   41.511
C   10.429   56.701   40.203
C   8.043   60.862   41.042
C   4.184   58.468   42.575
N   8.376   55.650   40.866
C   7.771   54.426   41.035
C   8.696   53.359   40.540
C   10.080   54.083   40.425
C   9.675   55.558   40.513
C   11.100   53.695   41.445
C   8.287   52.647   39.197
C   8.532   51.141   39.038
C   8.742   50.496   37.684
O   9.791   50.304   37.074
O   7.549   50.164   37.208
N   8.979   58.671   40.709
C   10.113   58.089   40.244
C   11.003   59.087   39.754
C   10.443   60.369   39.928
C   9.099   59.977   40.612
C   12.380   58.745   39.265
C   11.040   61.702   39.495
O   12.202   61.850   39.185
C   10.105   62.825   39.604
N   6.185   59.443   41.710
C   6.783   60.638   41.523
C   5.826   61.798   42.093
C   4.637   60.972   42.660
C   5.010   59.539   42.309
C   6.462   62.794   43.165
C   3.282   61.495   42.170
C   2.174   61.840   43.229
N   5.595   56.649   42.006
C   4.399   57.067   42.443
C   3.577   55.919   42.850
C   4.336   54.809   42.379
C   5.539   55.301   41.998
C   2.228   55.938   43.408
C   4.327   53.353   42.316
O   3.588   52.434   42.619
C   5.693   53.015   41.634
C   5.220   52.245   40.392
O   4.795   52.708   39.357
O   5.440   50.899   40.672
C   4.971   49.903   39.646
C   7.598   49.538   35.896
C   7.302   50.688   34.907
C   6.260   50.935   34.170
C   4.965   50.129   34.212
C   6.234   52.148   33.257
C   7.034   52.171   32.003
C   6.158   52.636   30.764
C   6.181   54.190   30.821
C   4.747   54.773   30.551
C   7.194   54.709   29.758
C   7.961   56.110   30.213
C   9.259   55.841   30.919
C   10.384   55.657   29.977
C   11.290   54.525   30.363
C   11.244   56.896   29.760
C   11.824   57.147   28.333
C   13.221   57.825   28.465
C   14.471   56.901   28.279
C   15.569   57.367   29.152
C   14.852   56.772   26.770
H   11.390   56.377   39.799
H   8.252   61.931   40.956
H   3.184   58.777   42.885
H   8.903   52.580   41.274
H   10.567   53.965   39.457
H   11.125   52.710   41.910
H   11.095   54.447   42.234
H   12.116   53.777   41.059
H   8.849   53.141   38.404
H   7.244   52.821   38.934
H   7.625   50.670   39.416
H   9.369   50.830   39.662
H   12.512   57.819   38.705
H   13.060   58.903   40.102
H   12.773   59.515   38.601
H   9.804   62.982   40.640
H   9.199   62.433   39.142
H   10.485   63.757   39.187
H   5.450   62.359   41.237
H   4.609   60.947   43.749
H   6.330   63.806   42.781
H   7.526   62.610   43.311
H   5.949   62.853   44.125
H   2.741   60.733   41.610
H   3.379   62.426   41.610
H   1.378   62.391   42.727
H   2.423   62.463   44.089
H   1.748   60.937   43.666
H   1.459   56.433   42.815
H   2.256   56.490   44.348
H   1.944   54.895   43.552
H   6.223   52.306   42.270
H   5.832   49.268   39.438
H   4.743   50.357   38.681
H   4.085   49.414   40.050
H   6.755   48.848   35.941
H   8.450   48.904   35.651
H   8.148   51.371   34.824
H   5.112   49.094   33.903
H   4.204   50.572   33.569
H   4.627   50.037   35.244
H   6.603   53.018   33.800
H   5.181   52.394   33.126
H   7.417   51.163   31.844
H   7.849   52.887   32.106
H   5.213   52.093   30.788
H   6.725   52.264   29.911
H   6.509   54.595   31.778
H   4.733   55.726   31.080
H   3.912   54.168   30.903
H   4.512   55.028   29.518
H   6.762   54.729   28.758
H   7.961   53.934   29.770
H   7.239   56.733   30.740
H   8.248   56.656   29.314
H   9.050   54.991   31.568
H   9.401   56.710   31.563
H   10.080   55.324   28.985
H   12.178   54.812   30.926
H   11.604   53.904   29.524
H   10.688   53.856   30.977
H   11.995   57.108   30.521
H   10.590   57.759   29.885
H   11.300   57.856   27.692
H   11.916   56.191   27.818
H   13.260   58.282   29.454
H   13.195   58.631   27.732
H   14.188   55.882   28.542
H   16.537   57.133   28.708
H   15.525   56.762   30.058
H   15.560   58.433   29.378
H   14.953   55.691   26.671
H   15.810   57.275   26.641
H   14.088   57.151   26.091
