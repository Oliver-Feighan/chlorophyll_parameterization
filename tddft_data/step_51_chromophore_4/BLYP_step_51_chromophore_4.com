%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_51_chromophore_4 TDDFT with blyp functional

0 1
Mg   9.253   4.179   27.664
C   10.532   2.209   30.416
C   7.630   6.135   29.842
C   7.755   5.518   24.973
C   10.339   1.352   25.613
N   9.056   4.164   29.843
C   9.874   3.423   30.723
C   9.636   3.992   32.184
C   8.518   5.118   31.990
C   8.359   5.124   30.436
C   7.209   4.953   32.752
C   10.892   4.537   32.804
C   10.766   5.370   34.098
C   11.528   5.026   35.350
O   12.521   5.646   35.793
O   11.109   3.791   35.883
N   7.899   5.659   27.397
C   7.290   6.327   28.437
C   6.395   7.311   27.897
C   6.453   7.236   26.443
C   7.380   6.102   26.214
C   5.642   8.320   28.764
C   5.673   7.932   25.308
O   5.769   7.790   24.126
C   4.739   9.101   25.546
N   9.026   3.463   25.599
C   8.423   4.275   24.679
C   8.470   3.603   23.250
C   9.412   2.407   23.443
C   9.603   2.376   24.957
C   7.068   3.051   22.840
C   10.724   2.618   22.724
C   10.936   1.776   21.523
N   10.205   2.226   27.914
C   10.612   1.287   26.976
C   11.256   0.170   27.594
C   11.309   0.513   28.961
C   10.622   1.760   29.121
C   11.890   -1.024   26.966
C   11.848   0.131   30.277
O   12.524   -0.790   30.598
C   11.331   1.214   31.264
C   10.635   0.577   32.349
O   9.652   -0.187   32.179
O   11.409   0.695   33.583
C   10.887   0.006   34.811
C   11.651   3.501   37.169
C   11.531   4.610   38.193
C   11.836   4.484   39.527
C   12.393   3.247   40.181
C   11.451   5.656   40.427
C   9.958   5.617   40.866
C   9.419   7.088   40.667
C   9.482   7.999   41.916
C   8.443   7.554   42.926
C   9.431   9.528   41.746
C   10.828   10.074   42.061
C   10.645   11.664   42.155
C   11.093   12.156   43.533
C   11.739   13.574   43.407
C   9.854   12.092   44.580
C   10.219   11.513   45.936
C   9.128   10.618   46.634
C   9.706   9.197   46.958
C   8.607   8.093   46.767
C   10.532   9.142   48.316
H   7.061   6.831   30.463
H   7.506   6.015   24.033
H   10.729   0.538   24.998
H   9.299   3.114   32.734
H   8.786   6.157   32.182
H   6.358   4.916   32.071
H   6.981   5.814   33.381
H   7.201   4.134   33.472
H   11.443   5.117   32.064
H   11.531   3.703   33.096
H   9.702   5.230   34.291
H   10.851   6.441   33.910
H   6.386   9.019   29.147
H   5.028   7.910   29.565
H   5.106   9.004   28.105
H   4.185   9.224   24.616
H   5.282   10.021   25.762
H   4.032   8.750   26.297
H   8.751   4.341   22.498
H   8.832   1.534   23.146
H   6.932   2.241   23.556
H   7.057   2.632   21.834
H   6.193   3.692   22.954
H   11.506   2.327   23.425
H   10.976   3.602   22.328
H   11.765   1.102   21.739
H   11.127   2.348   20.615
H   10.021   1.203   21.373
H   11.142   -1.741   26.628
H   12.521   -1.582   27.659
H   12.421   -0.857   26.029
H   12.103   1.811   31.751
H   10.305   0.828   35.229
H   11.723   -0.285   35.447
H   10.200   -0.829   34.676
H   12.709   3.272   37.040
H   11.125   2.680   37.656
H   10.961   5.462   37.821
H   11.664   2.820   40.870
H   13.296   3.621   40.664
H   12.767   2.395   39.615
H   11.762   6.597   39.975
H   12.031   5.567   41.345
H   9.784   5.215   41.864
H   9.437   5.008   40.127
H   8.391   7.050   40.306
H   9.967   7.569   39.857
H   10.425   7.801   42.426
H   7.859   6.687   42.620
H   7.738   8.322   43.245
H   9.070   7.375   43.800
H   8.615   10.058   42.237
H   9.459   9.786   40.687
H   11.559   9.822   41.293
H   11.083   9.625   43.020
H   9.671   12.126   41.994
H   11.416   12.057   41.492
H   11.933   11.507   43.780
H   11.972   14.121   44.321
H   11.132   14.322   42.898
H   12.727   13.426   42.972
H   9.079   11.475   44.124
H   9.393   13.060   44.772
H   10.364   12.371   46.592
H   11.226   11.101   46.009
H   8.437   10.461   45.806
H   8.725   11.171   47.482
H   10.428   8.958   46.177
H   7.809   8.370   47.456
H   8.849   7.056   47.002
H   8.126   8.163   45.791
H   11.594   8.980   48.131
H   9.983   8.410   48.909
H   10.388   10.121   48.772

