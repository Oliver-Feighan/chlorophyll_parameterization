%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_401_chromophore_4 TDDFT with PBE1PBE functional

0 1
Mg   8.684   3.391   27.331
C   10.241   1.962   30.111
C   7.142   5.651   29.527
C   6.677   4.584   24.744
C   9.869   0.889   25.319
N   8.545   3.607   29.616
C   9.424   3.021   30.534
C   9.178   3.572   31.926
C   8.135   4.787   31.734
C   7.920   4.687   30.156
C   6.845   4.704   32.673
C   10.532   4.081   32.568
C   10.360   4.981   33.802
C   11.197   4.547   34.999
O   12.427   4.492   34.944
O   10.439   4.323   36.117
N   7.220   4.971   27.134
C   6.809   5.856   28.161
C   5.904   6.833   27.588
C   5.782   6.544   26.184
C   6.605   5.351   25.926
C   5.248   7.956   28.390
C   4.860   7.211   25.182
O   4.472   6.622   24.174
C   4.162   8.479   25.426
N   8.306   2.871   25.320
C   7.464   3.496   24.420
C   7.612   2.947   22.975
C   8.697   1.814   23.218
C   8.940   1.769   24.745
C   6.200   2.499   22.393
C   10.061   2.036   22.431
C   10.240   1.184   21.232
N   9.783   1.762   27.516
C   10.314   0.888   26.615
C   11.377   0.100   27.164
C   11.310   0.454   28.540
C   10.318   1.448   28.721
C   12.210   -0.932   26.552
C   11.932   0.139   29.805
O   12.777   -0.750   30.118
C   11.296   1.146   30.855
C   10.669   0.553   32.050
O   9.573   -0.016   32.005
O   11.389   0.866   33.169
C   10.760   0.605   34.482
C   11.242   3.702   37.176
C   11.471   4.704   38.301
C   11.567   4.400   39.639
C   11.757   2.973   40.162
C   11.576   5.472   40.700
C   10.206   5.702   41.431
C   9.484   7.005   41.060
C   9.788   8.166   41.971
C   8.665   8.399   43.033
C   10.177   9.523   41.171
C   11.533   10.128   41.597
C   11.471   11.721   41.977
C   12.038   12.028   43.395
C   12.567   13.509   43.343
C   11.042   11.784   44.521
C   11.565   10.716   45.527
C   10.555   10.620   46.754
C   10.062   9.098   47.015
C   8.541   9.152   47.301
C   10.723   8.331   48.155
H   6.804   6.352   30.294
H   6.150   4.885   23.836
H   10.392   0.259   24.597
H   8.695   2.715   32.395
H   8.607   5.747   31.941
H   7.223   4.182   33.553
H   6.139   4.034   32.182
H   6.483   5.684   32.982
H   11.104   4.571   31.780
H   10.980   3.129   32.852
H   9.365   5.070   34.238
H   10.668   5.992   33.537
H   4.173   7.786   28.340
H   5.488   8.957   28.031
H   5.645   7.732   29.380
H   4.741   9.260   25.919
H   3.399   8.091   26.101
H   3.693   8.989   24.585
H   8.054   3.738   22.369
H   8.311   0.812   23.029
H   6.215   1.702   21.650
H   5.784   3.404   21.953
H   5.612   2.107   23.223
H   10.945   1.866   23.046
H   10.174   3.088   22.167
H   11.193   0.674   21.376
H   10.429   1.831   20.375
H   9.447   0.448   21.100
H   13.144   -1.208   27.041
H   12.514   -0.627   25.550
H   11.473   -1.735   26.591
H   12.110   1.829   31.097
H   9.726   0.298   34.326
H   10.806   1.525   35.065
H   11.357   -0.136   35.013
H   12.165   3.176   36.933
H   10.581   2.981   37.656
H   11.385   5.761   38.049
H   12.009   2.189   39.448
H   10.781   2.786   40.609
H   12.567   2.982   40.891
H   11.777   6.389   40.146
H   12.369   5.407   41.445
H   10.488   5.683   42.484
H   9.607   4.806   41.268
H   8.405   6.860   41.007
H   9.736   7.292   40.039
H   10.570   7.820   42.647
H   8.868   9.156   43.790
H   8.699   7.483   43.622
H   7.673   8.508   42.594
H   9.344   10.225   41.148
H   10.260   9.197   40.134
H   12.088   10.196   40.661
H   12.148   9.480   42.221
H   10.480   12.174   41.996
H   12.023   12.273   41.216
H   12.957   11.446   43.460
H   12.600   13.998   44.317
H   12.039   14.184   42.669
H   13.634   13.507   43.120
H   10.131   11.376   44.084
H   10.815   12.703   45.062
H   12.596   10.779   45.875
H   11.585   9.768   44.989
H   9.705   11.282   46.588
H   11.103   11.061   47.587
H   10.288   8.478   46.148
H   8.002   8.372   46.763
H   7.956   10.070   47.250
H   8.327   8.792   48.307
H   10.426   7.285   48.082
H   10.648   8.686   49.183
H   11.749   8.381   47.789

