%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1801_chromophore_15 TDDFT with cam-b3lyp functional

0 1
Mg   46.763   34.538   28.126
C   45.285   33.043   30.897
C   47.246   37.448   30.159
C   47.886   36.219   25.457
C   45.950   31.912   26.183
N   46.254   35.185   30.255
C   45.826   34.285   31.204
C   46.164   34.807   32.589
C   46.339   36.333   32.306
C   46.530   36.360   30.788
C   45.205   37.386   32.796
C   47.394   34.176   33.313
C   47.511   34.311   34.786
C   46.640   35.200   35.674
O   45.361   35.241   35.667
O   47.428   36.073   36.505
N   47.302   36.637   27.811
C   47.585   37.573   28.775
C   48.120   38.756   28.153
C   48.277   38.441   26.771
C   47.820   37.042   26.596
C   48.363   40.034   28.891
C   48.869   39.326   25.590
O   48.872   38.885   24.434
C   49.424   40.726   25.761
N   47.018   34.059   26.178
C   47.493   34.925   25.171
C   47.677   34.110   23.851
C   46.941   32.775   24.103
C   46.711   32.867   25.592
C   47.250   34.906   22.612
C   47.844   31.534   23.693
C   47.085   30.312   23.454
N   45.778   32.785   28.429
C   45.526   31.761   27.530
C   44.893   30.652   28.154
C   44.820   31.093   29.496
C   45.340   32.421   29.604
C   44.454   29.313   27.474
C   44.235   30.805   30.776
O   43.595   29.871   31.113
C   44.466   32.097   31.591
C   44.822   31.792   32.967
O   45.829   31.256   33.405
O   43.641   31.974   33.692
C   43.605   31.584   35.135
C   46.725   37.065   37.388
C   47.662   38.148   37.718
C   47.642   39.001   38.755
C   46.445   39.373   39.596
C   48.834   39.987   38.750
C   48.739   41.035   37.600
C   48.447   42.422   38.081
C   49.559   43.497   37.876
C   50.652   43.520   38.952
C   48.950   44.934   37.705
C   47.940   45.343   38.777
C   47.899   46.889   39.020
C   46.781   47.558   38.197
C   47.275   48.720   37.316
C   45.479   48.075   39.074
C   44.637   46.959   39.692
C   43.730   46.110   38.750
C   42.233   46.288   39.124
C   41.356   45.216   38.381
C   41.683   47.722   38.821
H   47.610   38.247   30.808
H   48.233   36.473   24.453
H   45.865   31.042   25.529
H   45.237   34.648   33.141
H   47.345   36.600   32.630
H   44.392   36.731   33.107
H   44.855   37.929   31.917
H   45.395   38.155   33.544
H   48.316   34.630   32.950
H   47.446   33.118   33.056
H   48.535   34.665   34.907
H   47.323   33.287   35.109
H   48.400   40.899   28.229
H   49.396   39.918   29.218
H   47.621   40.333   29.631
H   49.761   40.933   26.776
H   48.568   41.394   25.666
H   50.220   40.860   25.028
H   48.745   33.908   23.777
H   45.936   32.664   23.693
H   46.537   35.693   22.854
H   46.802   34.282   21.839
H   48.180   35.300   22.201
H   48.650   31.361   24.406
H   48.474   31.713   22.821
H   47.138   29.720   24.368
H   47.533   29.753   22.632
H   46.036   30.517   23.244
H   43.520   29.459   26.931
H   44.334   28.565   28.258
H   45.152   28.935   26.727
H   43.479   32.555   31.647
H   42.577   31.532   35.494
H   44.171   32.407   35.571
H   44.105   30.646   35.378
H   46.299   36.472   38.198
H   45.913   37.540   36.837
H   48.608   38.105   37.177
H   45.703   38.646   39.265
H   46.186   40.381   39.272
H   46.693   39.289   40.654
H   49.751   39.403   38.667
H   48.782   40.366   39.771
H   47.897   40.883   36.925
H   49.668   40.920   37.042
H   48.283   42.363   39.157
H   47.564   42.813   37.576
H   50.060   43.281   36.932
H   50.736   42.471   39.236
H   50.395   44.119   39.825
H   51.502   43.879   38.373
H   48.526   45.038   36.707
H   49.753   45.663   37.816
H   48.226   44.935   39.746
H   46.949   44.959   38.537
H   48.809   47.441   38.788
H   47.640   46.925   40.078
H   46.382   46.835   37.485
H   46.599   49.567   37.426
H   47.175   48.465   36.261
H   48.268   49.105   37.547
H   44.773   48.607   38.436
H   45.957   48.669   39.852
H   44.049   47.281   40.552
H   45.341   46.243   40.115
H   44.030   45.070   38.876
H   44.007   46.355   37.725
H   42.022   46.146   40.184
H   40.499   45.600   37.829
H   41.182   44.352   39.022
H   41.876   44.719   37.563
H   41.753   48.354   39.706
H   40.671   47.620   38.428
H   42.349   48.204   38.105

