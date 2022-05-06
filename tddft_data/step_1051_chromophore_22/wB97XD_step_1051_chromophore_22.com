%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1051_chromophore_22 TDDFT with wB97XD functional

0 1
Mg   8.993   47.666   25.135
C   6.736   47.756   27.930
C   11.446   48.523   27.384
C   11.095   47.508   22.691
C   6.271   47.976   23.023
N   9.073   48.090   27.347
C   8.041   48.036   28.282
C   8.547   48.261   29.693
C   9.843   49.085   29.423
C   10.229   48.506   27.969
C   9.585   50.613   29.412
C   8.746   46.973   30.628
C   8.753   47.244   32.172
C   8.616   46.055   33.175
O   8.551   44.854   32.810
O   8.324   46.508   34.442
N   11.095   47.920   25.021
C   11.965   48.265   26.046
C   13.310   48.268   25.544
C   13.209   47.801   24.172
C   11.802   47.734   23.854
C   14.452   48.904   26.368
C   14.313   47.545   23.218
O   14.098   47.155   22.108
C   15.756   47.670   23.531
N   8.652   47.963   23.113
C   9.791   47.658   22.290
C   9.352   47.509   20.859
C   7.853   48.024   20.925
C   7.507   48.022   22.392
C   10.170   48.153   19.738
C   6.811   47.177   20.054
C   6.018   47.957   18.991
N   6.909   47.677   25.325
C   5.934   47.790   24.386
C   4.594   47.493   25.002
C   4.865   47.471   26.343
C   6.285   47.654   26.531
C   3.303   47.424   24.293
C   4.229   47.415   27.639
O   3.049   47.277   27.910
C   5.397   47.598   28.721
C   5.296   46.418   29.609
O   5.477   45.302   29.291
O   5.080   46.853   30.853
C   5.081   45.884   31.891
C   8.199   45.426   35.386
C   9.188   45.599   36.492
C   9.027   45.407   37.837
C   7.703   45.029   38.561
C   10.198   45.766   38.799
C   10.159   45.259   40.187
C   11.581   45.162   40.831
C   11.460   45.491   42.357
C   11.598   47.017   42.510
C   12.581   44.696   43.106
C   12.179   43.264   43.536
C   13.108   42.103   43.118
C   13.625   41.250   44.324
C   13.825   39.732   43.966
C   14.927   41.927   44.734
C   15.141   41.969   46.272
C   15.078   40.647   47.015
C   15.837   40.466   48.354
C   16.814   39.242   48.396
C   14.932   40.276   49.513
H   12.141   48.721   28.202
H   11.760   47.268   21.859
H   5.400   48.050   22.369
H   7.757   48.915   30.061
H   10.486   48.913   30.287
H   8.551   50.727   29.086
H   10.248   51.025   28.651
H   9.777   51.135   30.350
H   9.680   46.514   30.302
H   8.001   46.233   30.334
H   7.890   47.885   32.350
H   9.652   47.858   32.238
H   14.040   49.104   27.357
H   14.820   49.851   25.974
H   15.230   48.147   26.469
H   15.947   46.909   24.287
H   15.741   48.661   23.987
H   16.355   47.520   22.633
H   9.427   46.447   20.625
H   7.825   49.042   20.535
H   10.888   48.846   20.177
H   9.573   48.696   19.005
H   10.730   47.376   19.217
H   6.113   46.757   20.779
H   7.519   46.508   19.565
H   6.144   47.439   18.041
H   6.290   49.000   18.828
H   4.935   47.870   19.073
H   3.371   47.226   23.223
H   2.879   48.398   24.537
H   2.844   46.543   24.743
H   5.219   48.452   29.375
H   4.114   45.383   31.832
H   5.117   46.448   32.823
H   5.892   45.157   31.931
H   8.381   44.411   35.035
H   7.228   45.469   35.880
H   10.093   46.104   36.153
H   6.898   44.934   37.833
H   7.544   45.833   39.279
H   7.746   44.063   39.063
H   10.237   46.855   38.845
H   11.127   45.543   38.274
H   9.755   44.247   40.186
H   9.516   45.842   40.847
H   12.274   45.806   40.290
H   11.983   44.149   40.817
H   10.556   45.279   42.928
H   10.841   47.494   41.889
H   12.652   47.255   42.363
H   11.372   47.176   43.564
H   12.843   45.345   43.942
H   13.472   44.689   42.478
H   11.202   43.116   43.076
H   12.053   43.346   44.616
H   13.914   42.480   42.488
H   12.493   41.463   42.485
H   12.937   41.308   45.167
H   13.017   39.107   44.348
H   14.719   39.260   44.375
H   13.687   39.632   42.890
H   14.959   42.966   44.408
H   15.807   41.482   44.268
H   14.567   42.830   46.614
H   16.155   42.311   46.478
H   15.401   39.902   46.289
H   14.002   40.475   47.053
H   16.360   41.393   48.588
H   17.768   39.677   48.101
H   16.555   38.461   47.681
H   17.084   38.782   49.347
H   15.405   40.697   50.400
H   14.773   39.229   49.775
H   13.983   40.797   49.386
