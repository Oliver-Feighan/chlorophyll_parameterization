%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_451_chromophore_1 TDDFT with cam-b3lyp functional

0 1
Mg   -2.361   17.201   27.000
C   -2.669   14.946   29.637
C   -3.248   19.843   29.042
C   -2.450   19.252   24.312
C   -2.416   14.455   24.747
N   -2.947   17.340   29.072
C   -2.887   16.308   30.025
C   -3.065   16.912   31.397
C   -3.664   18.383   31.100
C   -3.276   18.549   29.590
C   -5.254   18.490   31.483
C   -1.761   16.921   32.295
C   -1.967   16.505   33.763
C   -0.779   16.620   34.698
O   0.317   17.051   34.416
O   -1.082   16.070   35.945
N   -2.673   19.271   26.750
C   -2.931   20.211   27.740
C   -3.014   21.537   27.123
C   -2.665   21.365   25.765
C   -2.562   19.919   25.527
C   -3.513   22.774   27.990
C   -2.528   22.437   24.730
O   -2.379   22.251   23.544
C   -2.528   23.955   25.230
N   -2.415   16.921   24.820
C   -2.388   17.912   23.923
C   -2.526   17.321   22.518
C   -2.083   15.824   22.719
C   -2.257   15.684   24.211
C   -3.918   17.595   21.777
C   -0.715   15.586   22.052
C   0.592   15.996   22.821
N   -2.444   15.088   27.125
C   -2.404   14.142   26.116
C   -2.620   12.831   26.741
C   -2.660   13.035   28.120
C   -2.579   14.471   28.294
C   -2.886   11.483   26.047
C   -2.721   12.489   29.492
O   -2.773   11.351   29.887
C   -2.806   13.625   30.457
C   -1.808   13.369   31.542
O   -0.649   13.734   31.395
O   -2.490   12.888   32.611
C   -1.721   12.411   33.791
C   -0.071   15.912   36.984
C   -0.345   16.932   38.116
C   -0.310   16.703   39.458
C   -0.144   15.289   40.072
C   -0.640   17.708   40.534
C   0.456   18.726   40.899
C   0.142   20.166   40.558
C   0.185   21.088   41.767
C   -1.036   22.041   41.806
C   1.485   22.042   41.847
C   2.207   21.958   43.237
C   3.657   21.870   43.139
C   4.553   22.671   44.279
C   6.012   22.749   43.689
C   4.539   21.927   45.612
C   3.447   22.398   46.572
C   2.408   21.346   46.976
C   1.007   21.996   47.172
C   0.551   22.066   48.645
C   -0.166   21.403   46.219
H   -3.439   20.627   29.778
H   -2.274   19.857   23.420
H   -2.339   13.633   24.032
H   -3.842   16.271   31.813
H   -3.176   19.156   31.694
H   -5.562   17.485   31.770
H   -5.822   18.782   30.600
H   -5.418   19.164   32.324
H   -1.313   17.911   32.213
H   -1.031   16.256   31.834
H   -2.269   15.466   33.889
H   -2.771   17.156   34.105
H   -3.558   22.341   28.989
H   -4.401   23.206   27.528
H   -2.737   23.539   27.994
H   -2.344   24.517   24.314
H   -1.759   24.107   25.987
H   -3.487   24.156   25.706
H   -1.801   17.836   21.888
H   -2.918   15.342   22.211
H   -4.353   18.327   22.457
H   -4.508   16.685   21.670
H   -3.758   18.208   20.890
H   -0.767   16.137   21.113
H   -0.626   14.522   21.831
H   1.031   16.845   22.297
H   1.388   15.255   22.748
H   0.456   16.205   23.882
H   -1.910   11.096   25.755
H   -3.315   11.693   25.066
H   -3.493   10.824   26.668
H   -3.826   13.593   30.842
H   -1.995   11.380   34.014
H   -2.121   12.986   34.626
H   -0.636   12.510   33.815
H   0.938   16.152   36.648
H   -0.135   14.869   37.293
H   -0.922   17.777   37.741
H   0.217   14.548   39.359
H   -1.042   14.960   40.596
H   0.707   15.242   40.751
H   -0.844   17.154   41.450
H   -1.548   18.210   40.199
H   1.402   18.488   40.413
H   0.588   18.568   41.969
H   -0.793   20.303   40.015
H   0.937   20.445   39.867
H   0.070   20.524   42.692
H   -1.180   22.481   42.792
H   -1.984   21.545   41.597
H   -0.991   22.802   41.027
H   1.355   23.109   41.662
H   2.193   21.650   41.117
H   1.908   21.047   43.755
H   1.983   22.753   43.948
H   4.107   22.283   42.236
H   3.992   20.833   43.186
H   4.126   23.672   44.341
H   6.658   23.496   44.151
H   5.838   23.076   42.664
H   6.521   21.791   43.801
H   5.524   22.017   46.069
H   4.318   20.865   45.505
H   3.047   23.369   46.282
H   3.872   22.643   47.545
H   2.746   20.874   47.898
H   2.377   20.663   46.127
H   1.009   23.047   46.882
H   -0.507   21.818   48.728
H   0.644   23.117   48.920
H   1.139   21.450   49.324
H   -0.676   20.485   46.512
H   0.089   21.165   45.187
H   -0.950   22.156   46.296

