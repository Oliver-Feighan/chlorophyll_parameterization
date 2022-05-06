%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1401_chromophore_6 TDDFT with cam-b3lyp functional

0 1
Mg   17.198   -2.429   28.143
C   16.393   -0.194   30.806
C   19.168   -4.089   30.290
C   18.590   -3.860   25.599
C   15.692   0.044   26.021
N   17.879   -1.986   30.283
C   17.214   -1.253   31.247
C   17.525   -1.705   32.649
C   18.644   -2.763   32.367
C   18.521   -3.006   30.854
C   20.048   -2.329   32.819
C   16.303   -2.220   33.484
C   16.242   -1.672   34.922
C   17.492   -1.039   35.582
O   17.658   0.166   35.749
O   18.424   -1.981   35.980
N   18.568   -3.975   27.970
C   19.181   -4.545   28.967
C   19.826   -5.772   28.487
C   19.757   -5.690   27.066
C   18.946   -4.480   26.838
C   20.423   -6.870   29.356
C   20.198   -6.640   25.982
O   19.877   -6.466   24.810
C   20.856   -8.034   26.330
N   17.108   -1.985   26.146
C   17.834   -2.704   25.260
C   17.642   -2.238   23.742
C   16.681   -1.084   23.960
C   16.398   -1.006   25.441
C   19.047   -1.693   23.131
C   15.392   -1.052   23.081
C   14.125   -1.569   23.633
N   16.117   -0.514   28.318
C   15.460   0.294   27.384
C   14.851   1.362   28.031
C   15.197   1.230   29.365
C   15.996   0.095   29.483
C   13.950   2.521   27.455
C   15.054   1.840   30.634
O   14.448   2.902   31.011
C   15.673   0.850   31.658
C   16.595   1.691   32.537
O   17.690   2.229   32.318
O   15.881   1.985   33.636
C   16.303   3.086   34.448
C   19.565   -1.505   36.677
C   20.157   -2.753   37.276
C   20.539   -2.979   38.554
C   19.967   -2.197   39.630
C   21.336   -4.251   38.833
C   20.516   -5.479   39.093
C   20.567   -5.917   40.563
C   20.856   -7.421   40.603
C   19.560   -8.291   40.905
C   21.968   -7.864   41.523
C   23.224   -8.459   40.757
C   24.594   -8.303   41.367
C   25.463   -9.580   41.569
C   26.463   -9.258   42.707
C   26.263   -10.121   40.268
C   25.771   -11.465   39.716
C   25.991   -11.591   38.206
C   26.855   -12.803   37.745
C   26.266   -13.454   36.533
C   28.400   -12.529   37.662
H   19.762   -4.701   30.972
H   19.003   -4.291   24.684
H   15.236   0.769   25.344
H   17.789   -0.798   33.193
H   18.433   -3.696   32.889
H   20.200   -1.300   33.144
H   20.704   -2.576   31.984
H   20.403   -3.011   33.592
H   16.274   -3.303   33.606
H   15.440   -2.035   32.844
H   16.008   -2.441   35.658
H   15.431   -0.949   34.848
H   20.567   -6.590   30.399
H   21.401   -7.064   28.913
H   19.710   -7.692   29.285
H   20.170   -8.623   26.939
H   21.848   -7.860   26.748
H   20.982   -8.611   25.414
H   17.157   -3.038   23.183
H   17.194   -0.138   23.784
H   18.921   -0.767   22.571
H   19.364   -2.351   22.322
H   19.933   -1.628   23.762
H   15.543   -1.725   22.237
H   15.186   -0.076   22.641
H   14.349   -2.458   24.223
H   13.392   -1.818   22.866
H   13.703   -0.916   24.398
H   12.932   2.284   27.145
H   14.533   3.024   26.684
H   13.861   3.322   28.189
H   14.907   0.406   32.294
H   15.992   4.035   34.012
H   17.393   3.092   34.438
H   15.805   2.953   35.408
H   19.373   -0.689   37.374
H   20.316   -1.286   35.919
H   20.604   -3.408   36.528
H   19.715   -2.936   40.391
H   19.093   -1.602   39.364
H   20.623   -1.392   39.962
H   22.194   -3.956   39.436
H   21.842   -4.488   37.897
H   20.846   -6.166   38.314
H   19.470   -5.205   38.953
H   19.596   -5.771   41.037
H   21.365   -5.417   41.111
H   21.151   -7.770   39.613
H   18.743   -7.623   40.634
H   19.595   -8.552   41.963
H   19.521   -9.160   40.249
H   21.653   -8.548   42.310
H   22.403   -7.011   42.044
H   23.297   -7.917   39.814
H   22.974   -9.467   40.425
H   24.451   -7.772   42.308
H   25.098   -7.606   40.697
H   24.835   -10.396   41.928
H   27.496   -9.542   42.507
H   26.112   -9.916   43.502
H   26.358   -8.255   43.122
H   27.336   -10.219   40.433
H   26.253   -9.297   39.554
H   24.742   -11.441   40.073
H   26.300   -12.306   40.165
H   26.506   -10.733   37.772
H   25.017   -11.613   37.718
H   26.840   -13.495   38.587
H   25.569   -14.265   36.739
H   27.080   -13.865   35.936
H   25.751   -12.693   35.946
H   28.986   -13.312   38.142
H   28.691   -11.547   38.034
H   28.685   -12.481   36.611
