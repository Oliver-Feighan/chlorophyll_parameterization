%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1451_chromophore_3 ZINDO

0 1
Mg   2.042   7.531   26.417
C   2.429   9.424   29.283
C   2.088   4.700   28.229
C   2.499   5.903   23.515
C   2.342   10.574   24.521
N   2.245   7.145   28.537
C   2.256   8.003   29.539
C   2.163   7.336   30.916
C   2.305   5.798   30.527
C   2.135   5.864   28.998
C   3.570   5.104   31.066
C   0.843   7.725   31.599
C   0.925   8.219   33.073
C   2.029   7.644   33.988
O   2.928   8.294   34.486
O   1.782   6.329   34.364
N   2.247   5.573   25.945
C   2.160   4.514   26.817
C   2.322   3.301   26.071
C   2.618   3.622   24.670
C   2.547   5.114   24.681
C   2.399   2.000   26.879
C   2.973   2.709   23.475
O   3.309   3.181   22.382
C   2.798   1.189   23.571
N   2.259   8.119   24.313
C   2.413   7.229   23.306
C   2.328   7.924   21.891
C   2.129   9.430   22.249
C   2.227   9.404   23.858
C   3.536   7.622   20.935
C   0.733   9.869   21.765
C   0.636   11.399   21.453
N   2.323   9.547   26.765
C   2.437   10.654   25.913
C   2.603   11.871   26.665
C   2.547   11.432   28.006
C   2.411   10.018   27.996
C   2.664   13.133   26.040
C   2.590   11.881   29.428
O   2.913   12.979   29.898
C   2.269   10.600   30.264
C   3.162   10.639   31.440
O   4.342   10.308   31.548
O   2.554   11.361   32.435
C   3.245   11.632   33.684
C   2.618   5.719   35.341
C   1.828   4.577   36.003
C   2.000   4.183   37.298
C   3.135   4.506   38.342
C   0.787   3.315   37.736
C   1.013   1.872   37.993
C   -0.183   0.988   37.443
C   -0.705   0.005   38.570
C   -2.259   0.055   38.551
C   -0.205   -1.460   38.388
C   0.445   -1.885   39.693
C   1.973   -1.692   39.594
C   2.765   -3.030   39.713
C   4.102   -2.861   40.562
C   3.038   -3.620   38.327
C   3.341   -5.092   38.299
C   2.103   -6.042   38.214
C   2.093   -6.905   36.887
C   0.641   -7.434   36.509
C   3.007   -8.123   37.153
H   2.197   3.839   28.892
H   2.499   5.232   22.653
H   2.277   11.502   23.949
H   3.109   7.570   31.405
H   1.541   5.180   30.998
H   4.151   4.918   30.164
H   3.436   4.147   31.570
H   4.075   5.759   31.776
H   0.221   6.833   31.517
H   0.456   8.484   30.919
H   0.049   7.872   33.622
H   1.111   9.287   33.184
H   1.410   1.556   26.994
H   2.802   2.270   27.854
H   3.127   1.373   26.363
H   3.260   0.671   24.412
H   3.240   0.690   22.708
H   1.743   0.923   23.644
H   1.416   7.432   21.552
H   2.863   10.150   21.888
H   4.263   6.896   21.299
H   4.100   8.508   20.643
H   3.132   7.085   20.077
H   0.037   9.632   22.570
H   0.349   9.262   20.945
H   0.545   11.433   20.368
H   1.575   11.899   21.691
H   -0.254   11.787   21.947
H   2.662   14.027   26.664
H   1.830   13.109   25.339
H   3.597   13.219   25.483
H   1.259   10.781   30.631
H   3.425   10.780   34.339
H   2.659   12.288   34.327
H   4.217   12.091   33.501
H   3.042   6.350   36.123
H   3.367   5.314   34.660
H   1.027   4.206   35.364
H   2.683   4.863   39.267
H   3.712   5.390   38.074
H   3.827   3.697   38.575
H   -0.045   3.657   37.121
H   0.454   3.696   38.702
H   1.235   1.776   39.056
H   1.921   1.596   37.457
H   0.173   0.490   36.542
H   -0.949   1.645   37.030
H   -0.434   0.458   39.524
H   -2.645   -0.850   39.021
H   -2.761   0.086   37.584
H   -2.652   0.902   39.113
H   0.481   -1.544   37.545
H   -1.028   -2.144   38.181
H   0.225   -2.947   39.799
H   0.039   -1.430   40.596
H   2.170   -0.963   40.379
H   2.254   -1.395   38.583
H   2.126   -3.675   40.315
H   4.164   -1.906   41.084
H   4.979   -3.079   39.953
H   4.091   -3.618   41.346
H   3.765   -3.002   37.800
H   2.091   -3.523   37.797
H   3.900   -5.404   39.181
H   3.907   -5.168   37.370
H   1.188   -5.472   38.374
H   2.163   -6.711   39.073
H   2.493   -6.381   36.019
H   -0.154   -6.875   37.004
H   0.593   -8.468   36.850
H   0.518   -7.305   35.434
H   3.851   -7.781   37.752
H   3.237   -8.611   36.206
H   2.586   -8.891   37.801

