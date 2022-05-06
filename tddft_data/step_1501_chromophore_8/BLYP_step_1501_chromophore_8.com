%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1501_chromophore_8 TDDFT with blyp functional

0 1
Mg   43.804   3.338   47.107
C   41.836   6.311   46.863
C   40.713   1.608   46.836
C   45.495   0.801   46.721
C   46.697   5.502   46.694
N   41.512   3.997   46.975
C   40.998   5.288   46.975
C   39.465   5.231   47.116
C   39.168   3.619   47.303
C   40.545   3.023   47.112
C   38.575   3.187   48.646
C   38.690   5.929   45.952
C   39.363   6.091   44.558
C   38.781   5.357   43.349
O   38.377   4.227   43.376
O   38.811   6.128   42.240
N   43.183   1.479   46.648
C   41.937   0.915   46.656
C   42.124   -0.495   46.351
C   43.495   -0.748   46.204
C   44.142   0.509   46.506
C   40.932   -1.350   46.006
C   43.971   -2.130   45.857
O   43.182   -3.078   45.700
C   45.425   -2.393   45.805
N   45.760   3.222   46.802
C   46.250   1.998   46.792
C   47.764   1.949   47.132
C   48.156   3.447   46.530
C   46.812   4.125   46.696
C   48.026   1.867   48.641
C   48.654   3.427   45.008
C   50.181   3.587   44.675
N   44.262   5.455   47.051
C   45.439   6.166   46.754
C   45.263   7.540   46.651
C   43.842   7.653   46.756
C   43.278   6.353   46.919
C   46.284   8.565   46.426
C   42.733   8.628   46.703
O   42.798   9.856   46.514
C   41.387   7.768   46.785
C   40.746   8.191   48.072
O   41.217   8.131   49.164
O   39.544   8.868   47.816
C   38.760   9.620   48.895
C   38.230   5.630   40.992
C   39.440   5.183   40.114
C   40.067   5.977   39.199
C   39.959   7.518   39.161
C   41.280   5.424   38.516
C   41.113   4.479   37.355
C   41.994   3.221   37.304
C   41.344   2.083   36.393
C   41.197   0.742   37.145
C   41.980   2.092   34.974
C   41.629   3.362   34.203
C   42.917   4.223   34.004
C   43.571   4.170   32.616
C   45.112   4.303   32.829
C   42.962   5.145   31.657
C   41.899   4.583   30.704
C   42.047   5.154   29.318
C   43.068   4.321   28.477
C   42.370   3.202   27.638
C   44.143   5.176   27.767
H   39.794   1.025   46.752
H   46.079   -0.103   46.907
H   47.639   6.045   46.593
H   39.217   5.708   48.065
H   38.488   3.243   46.540
H   38.053   3.959   49.212
H   39.372   2.854   49.311
H   37.921   2.360   48.371
H   38.401   6.958   46.163
H   37.779   5.362   45.760
H   40.415   5.842   44.702
H   39.412   7.156   44.329
H   41.224   -1.797   45.056
H   40.040   -0.781   45.741
H   40.763   -2.080   46.797
H   45.977   -2.415   46.745
H   45.745   -1.469   45.323
H   45.720   -3.311   45.295
H   48.261   1.165   46.561
H   48.882   3.937   47.179
H   48.315   2.852   49.007
H   48.748   1.064   48.791
H   47.087   1.541   49.089
H   48.128   4.311   44.648
H   48.266   2.465   44.674
H   50.676   3.942   45.579
H   50.345   4.422   43.994
H   50.645   2.719   44.205
H   47.030   8.694   47.210
H   45.877   9.568   46.303
H   46.878   8.352   45.537
H   40.774   8.012   45.917
H   38.777   9.058   49.829
H   37.801   9.754   48.395
H   39.231   10.597   49.004
H   37.689   6.320   40.345
H   37.648   4.712   41.081
H   39.698   4.128   40.025
H   39.730   7.915   40.150
H   39.209   7.805   38.425
H   40.936   7.862   38.821
H   41.871   4.890   39.259
H   41.914   6.256   38.208
H   41.418   5.090   36.506
H   40.062   4.219   37.226
H   42.177   2.939   38.341
H   42.962   3.475   36.870
H   40.309   2.384   36.227
H   40.148   0.447   37.116
H   41.372   0.860   38.215
H   41.691   -0.118   36.694
H   41.599   1.260   34.381
H   43.059   2.109   35.123
H   40.828   3.897   34.713
H   41.246   3.115   33.213
H   43.655   4.058   34.789
H   42.681   5.273   34.179
H   43.274   3.166   32.314
H   45.516   4.858   33.676
H   45.581   4.664   31.914
H   45.507   3.292   32.925
H   43.723   5.611   31.031
H   42.345   5.901   32.143
H   40.918   4.905   31.056
H   41.853   3.494   30.722
H   42.375   6.191   29.399
H   41.132   5.361   28.763
H   43.659   3.719   29.167
H   42.513   2.197   28.035
H   42.885   3.249   26.678
H   41.292   3.332   27.549
H   45.069   4.613   27.651
H   44.346   6.115   28.282
H   43.786   5.428   26.769
