%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1301_chromophore_6 TDDFT with wB97XD functional

0 1
Mg   17.514   -2.574   27.440
C   16.676   -0.413   30.150
C   19.297   -4.588   29.751
C   18.736   -4.274   24.842
C   16.061   -0.247   25.305
N   17.924   -2.538   29.748
C   17.528   -1.482   30.568
C   17.924   -1.837   32.037
C   18.860   -3.117   31.876
C   18.716   -3.438   30.370
C   20.334   -2.984   32.426
C   16.736   -1.999   33.010
C   16.685   -1.225   34.294
C   18.030   -0.895   35.006
O   18.575   0.184   34.951
O   18.572   -2.009   35.656
N   18.709   -4.259   27.300
C   19.280   -4.998   28.392
C   19.846   -6.234   27.832
C   19.890   -6.007   26.406
C   19.158   -4.784   26.136
C   20.426   -7.230   28.709
C   20.636   -6.805   25.358
O   20.500   -6.543   24.166
C   21.452   -7.948   25.771
N   17.665   -2.172   25.399
C   18.165   -3.088   24.515
C   17.847   -2.685   23.088
C   16.879   -1.518   23.268
C   16.871   -1.233   24.783
C   18.986   -2.594   22.163
C   15.489   -1.701   22.625
C   14.373   -2.126   23.588
N   16.443   -0.803   27.638
C   15.886   0.024   26.680
C   15.084   1.061   27.323
C   15.400   0.903   28.674
C   16.250   -0.216   28.779
C   14.293   2.138   26.757
C   15.256   1.532   29.987
O   14.617   2.534   30.259
C   16.085   0.668   31.041
C   16.975   1.517   31.789
O   18.163   1.708   31.612
O   16.275   2.005   32.883
C   17.028   3.108   33.555
C   19.859   -1.844   36.375
C   20.016   -2.727   37.536
C   20.827   -3.728   37.760
C   21.854   -4.332   36.809
C   20.826   -4.486   39.146
C   19.836   -5.639   39.313
C   20.350   -6.745   40.315
C   20.519   -8.137   39.644
C   19.257   -9.038   39.758
C   21.727   -8.825   40.317
C   23.037   -8.516   39.677
C   23.996   -9.746   39.612
C   25.450   -9.378   39.956
C   26.480   -10.112   39.123
C   25.741   -9.449   41.506
C   25.645   -10.867   42.022
C   26.545   -11.194   43.204
C   27.943   -11.737   42.683
C   29.128   -10.919   43.185
C   28.133   -13.201   43.046
H   19.853   -5.208   30.458
H   18.859   -4.967   24.007
H   15.455   0.435   24.706
H   18.594   -1.054   32.393
H   18.411   -3.928   32.450
H   20.386   -3.788   33.160
H   20.401   -2.012   32.915
H   21.103   -3.156   31.673
H   16.532   -3.056   33.181
H   15.920   -1.674   32.365
H   16.239   -1.981   34.940
H   15.922   -0.457   34.176
H   20.070   -8.189   28.331
H   20.028   -7.208   29.723
H   21.509   -7.111   28.738
H   20.746   -8.677   26.171
H   22.045   -7.602   26.618
H   21.922   -8.393   24.894
H   17.358   -3.546   22.632
H   17.464   -0.708   22.834
H   19.107   -1.628   21.673
H   18.898   -3.368   21.400
H   19.851   -2.728   22.812
H   15.598   -2.551   21.953
H   15.166   -0.787   22.127
H   14.695   -2.843   24.344
H   13.636   -2.764   23.100
H   13.821   -1.222   23.843
H   14.058   1.821   25.741
H   15.009   2.952   26.644
H   13.387   2.317   27.335
H   15.310   0.304   31.715
H   17.328   3.971   32.962
H   17.913   2.782   34.103
H   16.349   3.376   34.364
H   20.125   -0.839   36.702
H   20.633   -2.149   35.671
H   19.587   -2.417   38.489
H   21.929   -3.815   35.852
H   21.558   -5.370   36.661
H   22.810   -4.111   37.282
H   20.728   -3.706   39.901
H   21.836   -4.827   39.375
H   19.633   -6.064   38.330
H   18.891   -5.269   39.713
H   19.545   -6.747   41.050
H   21.273   -6.408   40.787
H   20.652   -8.012   38.569
H   18.326   -8.473   39.707
H   19.262   -9.807   40.531
H   19.252   -9.645   38.853
H   21.526   -9.896   40.314
H   21.685   -8.556   41.373
H   23.520   -7.705   40.222
H   22.936   -8.070   38.687
H   23.945   -10.299   38.674
H   23.680   -10.491   40.342
H   25.570   -8.313   39.758
H   26.606   -9.552   38.197
H   26.108   -11.098   38.843
H   27.420   -10.103   39.673
H   25.046   -8.804   42.043
H   26.768   -9.101   41.613
H   25.815   -11.559   41.197
H   24.654   -11.207   42.322
H   26.012   -11.926   43.810
H   26.718   -10.341   43.861
H   27.979   -11.838   41.598
H   29.427   -10.282   42.352
H   30.006   -11.511   43.444
H   28.876   -10.182   43.947
H   27.494   -13.809   42.406
H   27.714   -13.292   44.049
H   29.192   -13.455   43.068

