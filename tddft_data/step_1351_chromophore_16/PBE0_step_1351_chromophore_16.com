%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1351_chromophore_16 TDDFT with PBE1PBE functional

0 1
Mg   40.463   41.583   26.886
C   39.327   43.935   29.328
C   41.222   39.433   29.531
C   41.725   39.592   24.715
C   40.071   44.106   24.438
N   40.298   41.681   29.191
C   39.759   42.776   29.930
C   39.758   42.355   31.402
C   40.218   40.873   31.484
C   40.707   40.635   30.006
C   41.172   40.547   32.620
C   38.361   42.697   32.124
C   37.956   41.848   33.361
C   37.286   42.589   34.546
O   36.363   43.424   34.447
O   37.743   42.098   35.737
N   41.299   39.669   27.106
C   41.578   38.968   28.233
C   42.072   37.636   27.890
C   42.149   37.649   26.438
C   41.742   38.990   26.015
C   42.475   36.595   28.877
C   42.449   36.538   25.445
O   42.574   36.806   24.258
C   42.724   35.039   25.954
N   40.767   41.809   24.831
C   41.316   40.782   24.132
C   41.462   41.097   22.661
C   40.720   42.464   22.557
C   40.476   42.832   24.012
C   42.872   41.052   21.923
C   39.454   42.523   21.646
C   38.202   41.807   22.186
N   39.931   43.602   26.861
C   39.781   44.425   25.773
C   39.221   45.669   26.283
C   39.090   45.520   27.647
C   39.457   44.200   27.989
C   38.961   46.919   25.499
C   38.707   46.261   28.809
O   38.400   47.434   28.973
C   38.809   45.248   29.973
C   39.657   45.754   30.975
O   40.859   46.025   30.827
O   38.840   46.036   32.045
C   39.542   46.642   33.196
C   37.327   42.998   36.854
C   37.979   42.369   38.115
C   37.641   42.687   39.400
C   36.579   43.611   39.877
C   38.394   42.033   40.557
C   37.527   41.360   41.656
C   38.238   40.007   42.034
C   38.215   39.739   43.570
C   39.558   40.096   44.250
C   37.760   38.293   43.987
C   36.215   38.332   44.309
C   35.423   37.269   43.482
C   34.678   36.174   44.235
C   34.322   34.924   43.476
C   33.478   36.758   44.925
C   33.348   36.217   46.409
C   32.032   35.511   46.733
C   31.427   36.146   48.013
C   30.152   36.948   47.749
C   31.160   35.053   49.038
H   41.159   38.583   30.213
H   42.265   39.035   23.946
H   39.966   44.899   23.696
H   40.543   43.012   31.773
H   39.251   40.370   31.484
H   42.141   40.259   32.213
H   40.727   39.734   33.194
H   41.305   41.395   33.292
H   37.639   42.566   31.318
H   38.403   43.724   32.488
H   38.792   41.361   33.862
H   37.175   41.197   32.967
H   41.636   35.905   28.790
H   42.490   37.035   29.874
H   43.455   36.182   28.636
H   42.830   34.329   25.134
H   42.014   34.767   26.736
H   43.704   35.113   26.425
H   40.872   40.360   22.116
H   41.352   43.244   22.134
H   43.720   41.036   22.607
H   43.007   41.846   21.188
H   42.865   40.063   21.464
H   39.645   42.069   20.673
H   39.114   43.550   21.510
H   37.353   42.452   21.959
H   38.313   41.502   23.227
H   38.025   40.910   21.591
H   39.384   46.823   24.499
H   39.317   47.725   26.142
H   37.925   47.192   25.299
H   37.849   44.948   30.394
H   38.987   47.521   33.525
H   40.557   46.963   32.962
H   39.409   45.936   34.015
H   36.242   42.937   36.937
H   37.661   44.032   36.765
H   38.661   41.545   37.909
H   36.994   44.483   40.382
H   35.872   43.121   40.547
H   36.054   43.995   39.003
H   39.096   42.726   41.020
H   38.972   41.228   40.103
H   36.497   41.065   41.456
H   37.529   42.033   42.514
H   39.278   40.066   41.710
H   37.776   39.215   41.444
H   37.544   40.431   44.078
H   40.087   39.171   44.481
H   39.426   40.664   45.171
H   40.176   40.710   43.595
H   38.312   37.897   44.839
H   37.951   37.631   43.143
H   35.684   39.233   44.001
H   36.011   38.119   45.358
H   36.164   36.671   42.951
H   34.715   37.794   42.841
H   35.383   35.794   44.974
H   33.382   34.983   42.927
H   34.116   34.040   44.081
H   35.105   34.665   42.764
H   32.563   36.403   44.452
H   33.437   37.844   44.846
H   33.612   37.029   47.086
H   34.172   35.509   46.496
H   32.190   34.433   46.719
H   31.329   35.568   45.903
H   32.072   36.859   48.527
H   29.323   36.718   48.419
H   29.797   36.795   46.730
H   30.278   37.999   48.008
H   30.968   35.624   49.947
H   32.004   34.394   49.241
H   30.375   34.415   48.633

