%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1201_chromophore_10 TDDFT with PBE1PBE functional

0 1
Mg   40.430   8.454   29.657
C   42.205   9.540   32.306
C   38.380   6.701   31.895
C   38.896   7.015   27.031
C   42.943   9.589   27.471
N   40.369   8.158   31.857
C   41.089   8.895   32.787
C   40.680   8.568   34.235
C   39.645   7.430   34.072
C   39.484   7.364   32.503
C   40.145   6.161   34.757
C   40.109   9.891   34.913
C   40.918   10.505   36.081
C   40.238   11.489   37.116
O   40.414   12.670   37.000
O   39.572   10.728   38.130
N   38.709   7.196   29.540
C   38.018   6.596   30.563
C   36.940   5.754   29.994
C   36.945   6.125   28.578
C   38.235   6.757   28.283
C   36.001   4.724   30.665
C   35.947   5.751   27.511
O   36.161   5.867   26.279
C   34.647   5.166   27.886
N   40.921   8.275   27.546
C   40.119   7.625   26.713
C   40.586   7.814   25.262
C   41.670   8.941   25.331
C   41.875   8.977   26.871
C   41.076   6.549   24.581
C   41.246   10.277   24.618
C   39.935   11.051   25.065
N   42.239   9.435   29.740
C   43.170   9.856   28.864
C   44.171   10.605   29.517
C   43.866   10.506   30.829
C   42.677   9.750   30.976
C   45.255   11.318   28.846
C   44.248   10.913   32.201
O   45.203   11.583   32.575
C   43.226   10.313   33.139
C   44.044   9.557   34.174
O   44.459   8.398   34.124
O   44.056   10.314   35.349
C   44.299   9.446   36.478
C   39.472   11.615   39.252
C   38.915   10.913   40.420
C   39.201   11.072   41.708
C   40.235   12.004   42.411
C   38.456   10.173   42.732
C   37.834   10.836   43.988
C   36.286   10.630   44.039
C   35.518   11.882   44.548
C   36.015   12.202   46.016
C   33.957   11.686   44.601
C   33.219   13.022   44.402
C   32.030   12.870   43.418
C   30.793   12.574   44.170
C   29.869   11.490   43.584
C   29.964   13.870   44.213
C   29.296   14.071   45.578
C   29.186   15.550   45.990
C   27.713   16.056   45.563
C   27.759   17.075   44.400
C   26.884   16.519   46.732
H   37.745   6.197   32.627
H   38.449   6.522   26.166
H   43.726   10.020   26.844
H   41.615   8.301   34.727
H   38.727   7.699   34.594
H   41.176   6.243   35.103
H   40.034   5.290   34.112
H   39.494   5.845   35.572
H   39.067   9.875   35.234
H   40.118   10.685   34.166
H   41.701   11.106   35.619
H   41.341   9.633   36.579
H   34.974   5.076   30.754
H   36.218   4.402   31.683
H   35.798   3.838   30.063
H   34.000   5.046   27.017
H   34.231   5.925   28.549
H   34.745   4.147   28.262
H   39.713   8.189   24.727
H   42.611   8.579   24.916
H   41.544   5.894   25.316
H   41.711   6.946   23.789
H   40.274   6.072   24.018
H   41.115   9.878   23.613
H   42.047   11.017   24.635
H   39.219   10.897   24.258
H   40.127   12.117   25.182
H   39.560   10.647   26.006
H   45.024   11.459   27.790
H   46.252   10.902   28.988
H   45.194   12.332   29.242
H   42.869   11.173   33.705
H   43.944   8.415   36.473
H   44.105   9.938   37.430
H   45.373   9.269   36.537
H   38.763   12.401   38.994
H   40.456   12.050   39.431
H   38.165   10.155   40.193
H   40.960   11.420   42.978
H   39.653   12.635   43.083
H   40.804   12.652   41.744
H   39.215   9.448   43.026
H   37.687   9.570   42.250
H   37.971   11.917   44.008
H   38.258   10.363   44.874
H   35.972   9.740   44.584
H   35.917   10.380   43.045
H   35.760   12.564   43.733
H   37.052   12.518   46.133
H   35.842   11.310   46.617
H   35.355   13.021   46.303
H   33.573   11.223   45.511
H   33.694   10.973   43.820
H   33.860   13.788   43.965
H   32.831   13.280   45.388
H   32.130   12.077   42.677
H   31.955   13.726   42.748
H   30.865   12.130   45.163
H   29.882   10.577   44.179
H   30.189   11.177   42.590
H   28.797   11.677   43.517
H   29.323   14.008   43.342
H   30.649   14.709   44.091
H   29.897   13.771   46.437
H   28.363   13.521   45.454
H   29.932   16.133   45.450
H   29.473   15.738   47.024
H   27.095   15.271   45.126
H   27.812   16.494   43.480
H   28.635   17.714   44.512
H   26.885   17.722   44.313
H   26.072   17.200   46.479
H   27.509   16.948   47.516
H   26.521   15.538   47.037

