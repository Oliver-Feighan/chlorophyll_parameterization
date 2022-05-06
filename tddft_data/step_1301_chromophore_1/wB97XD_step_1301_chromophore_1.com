%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1301_chromophore_1 TDDFT with wB97XD functional

0 1
Mg   -1.064   17.737   26.495
C   -1.426   15.702   29.381
C   -1.937   20.496   28.451
C   -1.398   19.552   23.707
C   -0.644   14.848   24.641
N   -1.562   18.025   28.694
C   -1.691   17.069   29.697
C   -1.988   17.674   31.039
C   -2.210   19.247   30.737
C   -1.879   19.301   29.191
C   -3.536   19.838   31.146
C   -0.943   17.375   32.196
C   -1.057   18.232   33.546
C   -0.796   17.441   34.854
O   -1.421   16.396   35.142
O   0.192   17.974   35.677
N   -1.485   19.755   26.152
C   -1.671   20.782   27.069
C   -1.770   22.080   26.407
C   -1.684   21.813   25.035
C   -1.480   20.343   24.893
C   -2.007   23.374   27.241
C   -1.862   22.814   23.959
O   -1.969   22.496   22.809
C   -1.901   24.255   24.245
N   -1.236   17.210   24.454
C   -1.290   18.185   23.445
C   -0.984   17.566   22.113
C   -0.474   16.074   22.502
C   -0.761   16.027   23.977
C   -2.163   17.558   21.076
C   0.960   15.674   22.133
C   2.101   16.572   22.561
N   -0.975   15.721   26.873
C   -0.845   14.662   25.994
C   -0.866   13.403   26.699
C   -0.994   13.795   28.018
C   -1.149   15.177   28.098
C   -0.636   11.926   26.125
C   -0.916   13.275   29.356
O   -0.601   12.149   29.798
C   -1.213   14.510   30.347
C   -0.032   14.403   31.333
O   1.144   14.651   31.083
O   -0.497   13.755   32.448
C   0.588   13.505   33.414
C   0.622   17.252   36.897
C   0.505   18.138   38.108
C   0.495   17.699   39.377
C   0.635   16.188   39.857
C   0.394   18.779   40.419
C   1.754   19.041   41.117
C   2.684   19.913   40.263
C   2.969   21.278   40.966
C   2.848   22.431   39.929
C   4.427   21.328   41.557
C   4.543   22.288   42.673
C   4.818   21.546   44.019
C   5.921   22.326   44.850
C   7.377   21.678   44.773
C   5.407   22.419   46.321
C   4.182   23.369   46.403
C   2.963   22.580   47.048
C   1.772   23.532   47.482
C   1.282   23.248   48.862
C   0.644   23.357   46.484
H   -2.236   21.365   29.041
H   -1.377   20.114   22.771
H   -0.328   13.952   24.101
H   -2.853   17.156   31.456
H   -1.389   19.710   31.285
H   -4.007   20.658   30.604
H   -3.295   20.281   32.112
H   -4.307   19.068   31.144
H   0.036   17.388   31.717
H   -1.253   16.376   32.500
H   -2.018   18.743   33.599
H   -0.198   18.891   33.417
H   -1.260   24.130   27.000
H   -1.944   23.248   28.322
H   -3.010   23.699   26.967
H   -2.853   24.514   24.709
H   -1.874   24.787   23.294
H   -0.957   24.460   24.751
H   -0.157   18.099   21.643
H   -1.115   15.393   21.942
H   -2.970   18.056   21.615
H   -2.428   16.543   20.780
H   -1.905   17.981   20.105
H   0.891   15.543   21.053
H   1.109   14.696   22.591
H   1.835   17.324   23.304
H   2.491   17.119   21.702
H   2.946   16.074   23.035
H   0.213   11.974   25.443
H   -1.532   11.430   25.750
H   -0.339   11.419   27.043
H   -2.170   14.396   30.855
H   1.214   12.659   33.130
H   0.128   13.336   34.388
H   1.131   14.421   33.644
H   1.670   16.950   36.895
H   0.093   16.312   37.058
H   0.376   19.198   37.889
H   1.585   16.073   40.379
H   0.855   15.528   39.018
H   -0.291   15.821   40.300
H   -0.376   18.464   41.123
H   -0.098   19.631   39.948
H   2.249   18.094   41.331
H   1.518   19.457   42.096
H   2.127   20.134   39.352
H   3.559   19.461   39.797
H   2.265   21.353   41.794
H   3.368   22.371   38.973
H   3.093   23.362   40.439
H   1.815   22.667   39.673
H   5.122   21.693   40.800
H   4.585   20.286   41.835
H   3.687   22.952   42.788
H   5.327   22.966   42.336
H   5.088   20.499   43.880
H   3.879   21.551   44.573
H   5.982   23.360   44.509
H   8.076   22.275   44.188
H   7.291   20.697   44.306
H   7.878   21.539   45.731
H   6.118   22.681   47.103
H   5.107   21.399   46.562
H   3.986   23.830   45.435
H   4.418   24.132   47.145
H   3.273   21.961   47.889
H   2.561   21.823   46.374
H   2.065   24.581   47.513
H   1.843   22.533   49.464
H   0.322   22.739   48.782
H   1.172   24.181   49.414
H   -0.232   23.836   46.921
H   0.465   22.305   46.264
H   0.861   23.852   45.537
