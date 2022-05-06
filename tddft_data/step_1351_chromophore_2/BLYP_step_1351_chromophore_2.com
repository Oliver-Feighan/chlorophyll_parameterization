%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1351_chromophore_2 TDDFT with blyp functional

0 1
Mg   2.708   0.642   44.303
C   5.946   2.061   44.170
C   1.503   3.374   42.581
C   -0.059   -1.137   43.913
C   4.326   -2.290   45.705
N   3.666   2.309   43.208
C   4.938   2.755   43.369
C   5.153   4.018   42.585
C   3.692   4.652   42.523
C   2.835   3.405   42.849
C   3.460   5.866   43.449
C   5.885   4.074   41.164
C   5.798   2.777   40.274
C   6.790   2.553   39.172
O   7.993   2.810   39.286
O   6.101   2.119   38.006
N   0.942   1.116   43.344
C   0.606   2.331   42.770
C   -0.792   2.265   42.411
C   -1.289   1.025   42.958
C   -0.141   0.224   43.374
C   -1.441   3.452   41.741
C   -2.758   0.581   42.974
O   -3.524   1.328   42.383
C   -3.185   -0.783   43.439
N   2.263   -1.484   44.680
C   0.976   -1.912   44.385
C   0.779   -3.365   44.875
C   2.195   -3.775   45.429
C   2.967   -2.443   45.300
C   -0.472   -3.725   45.694
C   2.875   -5.002   44.729
C   3.598   -5.973   45.692
N   4.707   0.114   44.926
C   5.148   -1.109   45.560
C   6.535   -0.929   45.900
C   6.887   0.314   45.414
C   5.765   0.909   44.816
C   7.420   -1.982   46.512
C   7.939   1.246   45.247
O   9.062   1.258   45.717
C   7.403   2.380   44.351
C   7.781   3.699   44.919
O   8.839   4.365   44.814
O   6.675   4.194   45.539
C   6.733   5.659   46.004
C   6.984   1.947   36.846
C   6.220   1.349   35.690
C   5.172   1.922   34.970
C   4.578   3.301   35.138
C   4.640   1.238   33.732
C   3.107   0.848   33.780
C   2.168   1.463   32.736
C   1.164   2.496   33.376
C   -0.084   1.710   33.843
C   0.824   3.610   32.434
C   0.457   3.201   30.970
C   -1.001   3.509   30.550
C   -0.975   4.612   29.380
C   -0.649   5.929   29.940
C   -2.283   4.641   28.594
C   -2.285   3.539   27.521
C   -2.149   4.122   26.069
C   -3.209   3.455   25.076
C   -2.831   3.652   23.599
C   -4.591   4.151   25.249
H   1.094   4.318   42.215
H   -0.929   -1.791   43.828
H   4.778   -3.097   46.284
H   5.849   4.611   43.177
H   3.484   4.778   41.460
H   2.934   5.572   44.357
H   2.889   6.551   42.822
H   4.371   6.368   43.776
H   6.912   4.298   41.454
H   5.439   4.850   40.542
H   4.828   2.938   39.803
H   5.646   1.894   40.895
H   -2.081   3.104   40.931
H   -0.694   4.065   41.238
H   -2.037   4.071   42.412
H   -3.070   -0.921   44.515
H   -2.721   -1.560   42.831
H   -4.255   -0.894   43.264
H   0.681   -4.013   44.004
H   2.114   -4.043   46.483
H   -1.302   -3.018   45.715
H   -0.198   -4.059   46.695
H   -0.838   -4.636   45.220
H   3.630   -4.555   44.083
H   2.219   -5.638   44.134
H   4.674   -5.900   45.528
H   3.249   -6.999   45.582
H   3.411   -5.727   46.738
H   7.309   -1.838   47.587
H   8.463   -1.802   46.250
H   7.092   -2.998   46.293
H   7.950   2.279   43.413
H   5.887   5.837   46.668
H   6.625   6.419   45.230
H   7.624   5.748   46.626
H   7.855   1.364   37.144
H   7.368   2.901   36.484
H   6.622   0.399   35.338
H   4.866   3.839   34.234
H   4.987   3.910   35.944
H   3.494   3.215   35.206
H   5.207   0.307   33.714
H   5.024   1.806   32.885
H   2.724   0.850   34.801
H   3.094   -0.222   33.572
H   1.685   0.633   32.220
H   2.758   1.961   31.967
H   1.644   2.822   34.299
H   -0.417   1.989   34.843
H   0.126   0.641   33.850
H   -0.831   1.877   33.068
H   1.720   4.230   32.418
H   0.085   4.285   32.865
H   0.652   2.175   30.656
H   1.173   3.799   30.405
H   -1.691   3.709   31.370
H   -1.502   2.619   30.171
H   -0.166   4.231   28.758
H   0.021   5.923   30.800
H   -1.558   6.448   30.244
H   -0.187   6.438   29.094
H   -2.550   5.613   28.179
H   -3.073   4.420   29.311
H   -3.199   2.966   27.673
H   -1.466   2.855   27.745
H   -1.172   3.823   25.689
H   -2.242   5.207   26.020
H   -3.144   2.412   25.387
H   -3.256   2.840   23.009
H   -1.747   3.675   23.491
H   -3.203   4.591   23.189
H   -5.147   3.455   25.878
H   -5.106   4.396   24.320
H   -4.449   5.052   25.846

