%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1651_chromophore_20 TDDFT with cam-b3lyp functional

0 1
Mg   6.924   57.004   41.410
C   5.707   53.564   40.851
C   9.801   56.034   39.707
C   7.672   60.191   41.142
C   3.515   57.808   42.285
N   7.533   55.064   40.200
C   7.017   53.768   40.247
C   8.065   52.698   39.728
C   9.377   53.572   39.758
C   8.884   54.976   39.891
C   10.169   53.331   41.057
C   7.548   52.102   38.404
C   8.402   52.191   37.179
C   8.132   51.091   36.099
O   8.407   49.915   36.146
O   7.509   51.647   34.930
N   8.549   57.956   40.637
C   9.625   57.439   40.041
C   10.552   58.461   39.682
C   10.120   59.617   40.298
C   8.743   59.335   40.725
C   11.782   58.191   38.843
C   10.856   61.004   40.351
O   11.998   61.180   39.855
C   10.190   62.286   40.902
N   5.726   58.783   41.759
C   6.358   59.977   41.628
C   5.373   61.076   41.804
C   3.999   60.312   42.185
C   4.405   58.860   42.052
C   5.791   62.129   42.808
C   2.849   60.668   41.180
C   1.891   61.745   41.672
N   5.026   55.968   41.773
C   3.756   56.404   42.081
C   2.812   55.290   42.305
C   3.574   54.135   41.850
C   4.883   54.585   41.500
C   1.434   55.343   42.762
C   3.443   52.735   41.532
O   2.487   51.967   41.711
C   4.787   52.322   40.818
C   4.559   51.710   39.492
O   4.047   52.235   38.481
O   5.236   50.512   39.487
C   5.010   49.743   38.283
C   7.170   50.671   33.955
C   6.509   51.305   32.655
C   5.543   50.784   31.877
C   4.803   49.455   32.195
C   5.118   51.554   30.682
C   6.198   52.116   29.728
C   5.790   53.511   29.334
C   6.256   54.547   30.329
C   5.213   55.656   30.301
C   7.667   55.029   29.925
C   8.646   55.271   31.153
C   10.075   54.941   30.692
C   11.095   56.159   30.509
C   10.616   57.244   29.541
C   12.448   55.569   30.163
C   13.623   56.456   30.726
C   14.208   57.605   29.785
C   15.698   57.849   30.010
C   16.105   59.374   30.130
C   16.426   57.078   28.829
H   10.743   55.592   39.377
H   7.832   61.260   40.988
H   2.463   58.061   42.432
H   7.931   51.952   40.512
H   10.008   53.641   38.871
H   9.712   52.717   41.834
H   10.437   54.260   41.560
H   11.133   52.948   40.724
H   6.635   52.620   38.109
H   7.360   51.029   38.424
H   9.440   51.933   37.389
H   8.224   53.101   36.607
H   11.774   57.201   38.388
H   12.698   58.304   39.422
H   11.685   58.843   37.975
H   9.482   62.734   40.204
H   10.906   63.072   41.141
H   9.704   61.947   41.817
H   5.279   61.578   40.841
H   3.699   60.458   43.222
H   6.497   61.742   43.543
H   4.943   62.610   43.295
H   6.301   62.967   42.332
H   2.261   59.818   40.834
H   3.327   60.985   40.254
H   0.883   61.346   41.790
H   1.751   62.552   40.952
H   2.205   62.167   42.627
H   0.757   55.018   41.971
H   1.219   56.375   43.040
H   1.253   54.788   43.683
H   5.159   51.666   41.604
H   4.195   50.129   37.671
H   4.708   48.752   38.622
H   5.911   49.763   37.669
H   6.368   50.173   34.500
H   7.988   49.995   33.705
H   7.006   52.199   32.278
H   5.005   48.747   31.391
H   3.728   49.584   32.321
H   5.190   48.927   33.066
H   4.640   52.413   31.153
H   4.353   50.991   30.148
H   6.159   51.504   28.827
H   7.234   52.077   30.062
H   4.702   53.521   29.265
H   6.083   53.753   28.312
H   6.303   54.123   31.331
H   5.807   56.560   30.167
H   4.695   55.774   31.252
H   4.514   55.537   29.473
H   7.721   55.980   29.395
H   8.067   54.216   29.318
H   8.315   54.739   32.045
H   8.472   56.336   31.304
H   10.091   54.382   29.757
H   10.551   54.274   31.411
H   11.178   56.591   31.507
H   11.269   57.420   28.687
H   10.619   58.197   30.070
H   9.655   57.067   29.058
H   12.509   55.405   29.087
H   12.604   54.619   30.674
H   14.382   55.744   31.053
H   13.428   57.032   31.630
H   13.592   58.469   30.034
H   13.844   57.336   28.794
H   16.086   57.542   30.980
H   16.263   59.969   29.230
H   17.000   59.438   30.748
H   15.314   59.663   30.822
H   16.822   57.746   28.064
H   15.766   56.428   28.255
H   17.228   56.524   29.316

