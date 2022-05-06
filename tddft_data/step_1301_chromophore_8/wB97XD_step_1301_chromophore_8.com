%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1301_chromophore_8 TDDFT with wB97XD functional

0 1
Mg   44.915   1.892   47.797
C   42.884   4.528   46.687
C   42.427   -0.297   47.370
C   47.073   -0.652   48.397
C   47.702   4.141   47.458
N   42.929   2.116   47.123
C   42.289   3.267   46.778
C   40.785   3.008   46.518
C   40.639   1.595   47.219
C   42.114   1.046   47.109
C   40.215   1.672   48.721
C   40.370   3.026   45.054
C   39.380   4.117   44.579
C   39.948   5.105   43.433
O   40.030   6.337   43.547
O   40.256   4.457   42.277
N   44.765   -0.208   47.898
C   43.610   -0.860   47.619
C   43.821   -2.276   47.680
C   45.259   -2.482   47.978
C   45.795   -1.056   48.086
C   42.761   -3.319   47.430
C   46.011   -3.863   48.159
O   45.465   -4.903   48.038
C   47.473   -3.960   48.481
N   47.124   1.682   47.637
C   47.694   0.587   48.233
C   49.116   0.779   48.531
C   49.326   2.253   47.997
C   47.937   2.764   47.690
C   49.476   0.485   50.018
C   50.267   2.307   46.777
C   51.791   2.805   47.034
N   45.269   3.932   47.212
C   46.400   4.793   47.212
C   46.003   6.224   47.073
C   44.591   6.116   46.915
C   44.225   4.752   46.872
C   46.837   7.432   46.964
C   43.430   6.927   46.687
O   43.166   8.121   46.613
C   42.289   5.956   46.717
C   41.424   6.177   47.884
O   41.879   6.051   49.045
O   40.115   6.521   47.569
C   39.135   6.651   48.656
C   40.318   5.255   41.027
C   41.518   4.937   40.252
C   42.235   5.748   39.450
C   41.754   7.227   39.225
C   43.626   5.291   38.986
C   43.733   4.444   37.684
C   44.415   5.165   36.574
C   44.518   4.380   35.256
C   45.938   3.795   35.154
C   44.165   5.183   34.046
C   43.040   4.541   33.177
C   43.230   5.134   31.703
C   44.051   4.325   30.725
C   45.455   4.910   30.546
C   43.366   4.166   29.336
C   43.885   2.916   28.458
C   45.087   3.215   27.468
C   44.982   2.537   25.996
C   45.275   3.513   24.818
C   45.823   1.228   25.872
H   41.542   -0.916   47.206
H   47.745   -1.429   48.766
H   48.465   4.905   47.619
H   40.205   3.782   47.019
H   40.027   0.912   46.630
H   40.972   1.262   49.391
H   39.310   1.064   48.742
H   39.908   2.672   49.027
H   39.896   2.088   44.766
H   41.292   3.155   44.488
H   39.129   4.718   45.454
H   38.405   3.797   44.211
H   43.104   -3.912   46.582
H   41.804   -2.860   47.182
H   42.624   -3.841   48.377
H   47.583   -3.727   49.541
H   47.924   -3.275   47.763
H   47.840   -4.980   48.368
H   49.699   0.094   47.917
H   49.673   2.892   48.809
H   49.863   -0.503   50.267
H   48.626   0.783   50.632
H   50.290   1.162   50.278
H   49.927   2.988   45.996
H   50.313   1.390   46.190
H   52.475   1.963   46.923
H   51.936   3.275   48.006
H   52.055   3.589   46.325
H   46.519   7.954   46.061
H   47.922   7.330   46.979
H   46.567   8.039   47.828
H   41.728   6.006   45.784
H   39.461   6.401   49.665
H   38.277   5.985   48.558
H   38.829   7.697   48.660
H   40.224   6.332   41.166
H   39.441   5.019   40.424
H   42.065   4.080   40.646
H   41.587   7.762   40.159
H   40.818   7.343   38.677
H   42.533   7.706   38.632
H   44.064   4.768   39.836
H   44.259   6.179   38.965
H   42.779   4.184   37.225
H   44.280   3.566   38.027
H   45.426   5.472   36.843
H   43.948   6.078   36.207
H   43.873   3.501   35.234
H   46.367   3.505   36.114
H   46.682   4.434   34.679
H   45.896   2.901   34.532
H   45.059   5.392   33.459
H   43.690   6.130   34.302
H   42.034   4.827   33.485
H   43.143   3.457   33.222
H   43.714   6.100   31.846
H   42.216   5.453   31.459
H   44.216   3.341   31.163
H   46.124   4.383   29.866
H   45.995   4.881   31.492
H   45.391   5.906   30.108
H   43.444   5.109   28.794
H   42.316   4.051   29.606
H   42.980   2.546   27.976
H   44.213   2.171   29.183
H   45.904   2.755   28.024
H   45.206   4.298   27.442
H   43.967   2.169   25.845
H   44.629   3.297   23.968
H   46.304   3.600   24.470
H   45.095   4.537   25.147
H   45.186   0.428   25.496
H   46.170   0.915   26.857
H   46.603   1.517   25.168
