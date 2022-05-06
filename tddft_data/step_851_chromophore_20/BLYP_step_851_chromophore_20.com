%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_851_chromophore_20 TDDFT with blyp functional

0 1
Mg   6.257   56.909   42.406
C   4.577   53.770   41.921
C   9.129   55.476   41.229
C   7.292   60.020   41.967
C   3.195   58.180   43.690
N   6.735   54.844   41.552
C   5.915   53.682   41.500
C   6.693   52.492   40.982
C   8.191   53.028   41.041
C   8.041   54.549   41.295
C   9.114   52.283   42.062
C   6.399   52.155   39.455
C   7.070   50.941   38.696
C   6.172   50.157   37.794
O   5.693   49.018   38.036
O   5.965   50.749   36.607
N   8.057   57.647   41.590
C   9.111   56.914   41.249
C   10.076   57.841   40.773
C   9.614   59.176   41.011
C   8.285   59.021   41.596
C   11.234   57.502   39.971
C   10.428   60.524   40.626
O   11.634   60.453   40.397
C   9.780   61.824   40.702
N   5.270   58.824   42.693
C   6.028   59.934   42.584
C   5.288   61.187   42.874
C   3.947   60.610   43.558
C   4.132   59.094   43.367
C   6.093   62.277   43.702
C   2.600   61.056   42.898
C   1.494   61.566   43.890
N   4.302   56.142   42.803
C   3.199   56.810   43.406
C   2.133   55.875   43.632
C   2.635   54.660   42.988
C   3.921   54.905   42.508
C   0.787   56.056   44.183
C   2.203   53.325   42.560
O   1.131   52.795   42.523
C   3.496   52.715   41.880
C   3.035   52.444   40.532
O   2.830   53.272   39.673
O   2.841   51.104   40.442
C   2.485   50.551   39.194
C   5.520   49.938   35.471
C   6.045   50.518   34.171
C   5.318   50.472   33.046
C   3.993   49.728   32.771
C   5.753   51.241   31.884
C   5.569   52.693   31.916
C   4.897   53.346   30.624
C   5.273   54.811   30.549
C   4.238   55.463   29.613
C   6.594   54.918   29.747
C   7.514   56.010   30.318
C   8.610   55.553   31.338
C   9.989   55.332   30.552
C   10.932   54.404   31.330
C   10.724   56.686   30.342
C   11.320   56.895   28.926
C   12.712   57.514   28.883
C   13.795   56.518   28.351
C   14.933   56.220   29.340
C   14.420   57.121   27.055
H   10.076   55.128   40.810
H   7.634   60.992   41.605
H   2.375   58.564   44.301
H   6.560   51.528   41.473
H   8.644   53.078   40.050
H   9.947   51.875   41.489
H   8.654   51.413   42.530
H   9.558   52.927   42.821
H   6.506   52.913   38.679
H   5.350   51.941   39.249
H   7.666   50.279   39.325
H   7.768   51.338   37.960
H   11.154   57.903   38.960
H   11.415   56.430   39.896
H   12.094   58.030   40.382
H   9.766   62.365   41.648
H   8.800   61.802   40.227
H   10.311   62.470   40.003
H   5.037   61.659   41.924
H   3.994   60.764   44.636
H   5.904   62.233   44.775
H   5.915   63.293   43.349
H   7.141   62.032   43.535
H   2.140   60.187   42.429
H   2.901   61.867   42.235
H   1.255   62.622   43.766
H   1.769   61.375   44.928
H   0.592   61.023   43.609
H   0.787   56.858   44.921
H   0.426   55.171   44.706
H   0.099   56.334   43.384
H   3.779   51.797   42.395
H   2.824   51.245   38.424
H   1.477   50.192   38.984
H   3.160   49.701   39.094
H   4.434   49.864   35.526
H   5.925   48.932   35.586
H   6.982   51.074   34.144
H   3.280   50.484   32.442
H   3.495   49.429   33.693
H   4.160   48.809   32.209
H   5.334   50.873   30.948
H   6.813   51.014   31.771
H   6.511   53.241   31.915
H   5.229   52.987   32.910
H   3.818   53.325   30.778
H   5.185   52.836   29.705
H   5.343   55.281   31.530
H   3.881   54.798   28.827
H   4.706   56.325   29.138
H   3.401   55.821   30.214
H   6.508   55.068   28.671
H   7.163   54.008   29.938
H   6.928   56.812   30.767
H   7.990   56.446   29.440
H   8.272   54.719   31.953
H   8.717   56.299   32.125
H   9.839   54.894   29.565
H   11.927   54.195   30.937
H   10.439   53.442   31.470
H   11.131   54.911   32.274
H   11.503   56.851   31.086
H   9.999   57.451   30.622
H   10.600   57.507   28.382
H   11.173   55.891   28.528
H   13.047   57.791   29.882
H   12.716   58.457   28.336
H   13.379   55.554   28.059
H   15.795   55.776   28.843
H   14.593   55.424   30.002
H   15.289   57.125   29.833
H   14.756   56.289   26.437
H   15.180   57.855   27.322
H   13.597   57.617   26.540
