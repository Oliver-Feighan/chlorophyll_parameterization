%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1851_chromophore_19 TDDFT with PBE1PBE functional

0 1
Mg   25.413   50.600   26.733
C   23.201   51.521   29.292
C   27.695   49.671   29.094
C   27.617   50.350   24.378
C   23.077   51.959   24.467
N   25.520   50.870   29.017
C   24.443   51.038   29.825
C   24.892   51.030   31.292
C   26.274   50.271   31.218
C   26.563   50.308   29.691
C   27.424   50.814   32.043
C   23.787   50.425   32.233
C   24.278   49.832   33.607
C   24.271   50.744   34.845
O   23.760   51.820   34.898
O   24.936   50.107   35.801
N   27.394   50.029   26.711
C   28.158   49.664   27.751
C   29.466   49.292   27.316
C   29.433   49.458   25.885
C   28.118   49.949   25.597
C   30.586   48.762   28.230
C   30.542   49.128   24.863
O   30.318   49.142   23.682
C   31.888   48.814   25.364
N   25.411   51.191   24.762
C   26.386   50.873   23.984
C   26.214   51.276   22.550
C   24.626   51.515   22.523
C   24.278   51.502   24.036
C   26.982   52.585   22.072
C   23.825   50.560   21.559
C   23.152   49.297   22.088
N   23.506   51.380   26.801
C   22.687   51.856   25.748
C   21.419   52.225   26.211
C   21.564   52.120   27.628
C   22.846   51.616   27.912
C   20.304   52.872   25.443
C   20.922   52.491   28.888
O   19.784   52.917   29.144
C   21.939   52.131   29.973
C   22.230   53.405   30.674
O   22.685   54.404   30.151
O   21.982   53.243   31.983
C   22.212   54.352   32.938
C   24.842   50.772   37.159
C   24.975   49.698   38.208
C   26.023   49.462   39.073
C   27.334   50.219   39.027
C   25.798   48.397   40.173
C   26.769   47.290   40.645
C   26.092   45.902   40.831
C   26.542   45.340   42.203
C   26.256   43.787   42.182
C   25.830   46.086   43.411
C   26.877   46.481   44.458
C   26.265   47.203   45.615
C   27.103   48.409   46.072
C   26.680   49.012   47.407
C   27.191   49.517   44.961
C   28.597   50.142   44.910
C   28.601   51.637   45.437
C   28.712   52.644   44.259
C   27.529   52.518   43.198
C   30.091   52.752   43.543
H   28.416   49.207   29.770
H   28.197   49.977   23.531
H   22.362   52.412   23.777
H   25.112   52.076   31.505
H   26.213   49.212   31.470
H   27.123   51.574   32.764
H   28.234   51.215   31.433
H   27.894   50.027   32.634
H   23.023   49.840   31.720
H   23.294   51.346   32.543
H   25.251   49.342   33.639
H   23.606   48.984   33.737
H   30.294   48.623   29.270
H   31.324   49.555   28.348
H   30.793   47.752   27.878
H   31.963   47.807   25.774
H   32.183   49.539   26.123
H   32.540   48.869   24.493
H   26.490   50.475   21.865
H   24.395   52.537   22.221
H   26.336   53.450   21.923
H   27.527   52.288   21.177
H   27.743   52.999   22.733
H   24.452   50.255   20.722
H   23.077   51.147   21.026
H   23.555   49.080   23.077
H   23.384   48.404   21.508
H   22.084   49.497   22.007
H   19.897   52.222   24.668
H   20.485   53.854   25.006
H   19.444   53.045   26.089
H   21.544   51.375   30.652
H   22.713   53.804   33.737
H   21.293   54.802   33.313
H   22.832   55.134   32.498
H   23.878   51.257   37.312
H   25.665   51.479   37.269
H   24.107   49.077   38.427
H   27.890   49.789   39.860
H   27.348   51.291   39.224
H   27.855   50.010   38.093
H   24.899   47.830   39.933
H   25.473   48.736   41.156
H   27.284   47.698   41.515
H   27.547   47.181   39.889
H   26.416   45.244   40.024
H   25.005   45.928   40.906
H   27.609   45.436   42.404
H   25.235   43.655   42.539
H   26.834   43.375   43.010
H   26.439   43.325   41.212
H   25.018   45.482   43.817
H   25.442   47.022   43.010
H   27.629   47.061   43.922
H   27.318   45.591   44.907
H   26.097   46.481   46.414
H   25.319   47.635   45.288
H   28.098   47.974   46.159
H   26.052   49.896   47.296
H   27.595   49.344   47.898
H   26.124   48.318   48.036
H   26.421   50.272   45.119
H   26.999   49.057   43.991
H   28.814   50.193   43.843
H   29.462   49.635   45.337
H   29.554   51.733   45.958
H   27.825   51.839   46.175
H   28.561   53.619   44.723
H   26.840   51.748   43.545
H   28.029   52.211   42.279
H   27.023   53.480   43.130
H   30.955   52.825   44.204
H   29.981   53.729   43.073
H   30.283   52.128   42.670

