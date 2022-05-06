%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1001_chromophore_15 TDDFT with blyp functional

0 1
Mg   45.932   34.142   28.004
C   44.512   32.062   30.500
C   45.672   36.834   30.129
C   47.179   36.186   25.653
C   45.958   31.400   25.879
N   45.109   34.384   30.049
C   44.840   33.384   30.928
C   44.876   33.900   32.364
C   44.884   35.495   32.174
C   45.214   35.588   30.706
C   43.655   36.290   32.636
C   46.066   33.254   33.239
C   46.968   34.183   33.978
C   46.580   34.675   35.380
O   45.454   34.738   35.809
O   47.723   35.264   35.982
N   46.323   36.198   27.936
C   46.152   37.095   28.908
C   46.663   38.407   28.532
C   47.212   38.311   27.226
C   46.896   36.856   26.871
C   46.602   39.555   29.464
C   47.911   39.437   26.287
O   48.233   39.252   25.101
C   48.077   40.816   26.860
N   46.336   33.905   25.950
C   46.896   34.888   25.248
C   47.458   34.308   23.865
C   47.125   32.763   23.984
C   46.419   32.666   25.337
C   46.877   34.940   22.625
C   48.281   31.805   23.901
C   48.282   31.000   22.649
N   45.286   32.179   28.050
C   45.446   31.173   27.167
C   44.835   29.946   27.686
C   44.541   30.193   29.016
C   44.775   31.598   29.176
C   44.669   28.532   26.967
C   44.098   29.597   30.306
O   43.922   28.446   30.634
C   43.819   30.841   31.225
C   44.143   30.495   32.638
O   45.092   29.836   33.043
O   43.158   31.077   33.411
C   43.277   30.894   34.838
C   47.573   35.668   37.359
C   47.653   37.151   37.502
C   46.726   38.058   38.007
C   45.342   37.673   38.651
C   47.111   39.542   38.071
C   47.622   40.010   39.450
C   48.917   40.865   39.171
C   48.551   42.318   38.704
C   49.119   42.630   37.217
C   48.798   43.456   39.725
C   47.526   44.343   39.913
C   47.847   45.792   40.238
C   47.032   46.746   39.361
C   47.750   48.010   39.057
C   45.608   46.924   39.946
C   44.634   47.059   38.784
C   43.716   45.825   38.645
C   42.227   46.119   38.564
C   41.791   46.326   40.087
C   41.440   45.046   37.973
H   45.812   37.528   30.960
H   47.741   36.827   24.971
H   45.868   30.592   25.149
H   43.942   33.622   32.851
H   45.749   35.916   32.688
H   43.865   36.859   33.542
H   42.783   35.648   32.762
H   43.428   37.110   31.955
H   46.662   32.571   32.632
H   45.694   32.641   34.060
H   47.514   34.938   33.411
H   47.798   33.508   34.187
H   47.596   39.994   29.386
H   46.305   39.263   30.471
H   45.842   40.306   29.247
H   47.104   41.274   27.036
H   48.770   41.388   26.244
H   48.658   40.793   27.782
H   48.543   34.380   23.937
H   46.343   32.481   23.279
H   47.322   35.930   22.528
H   45.834   35.178   22.831
H   47.051   34.266   21.786
H   48.257   31.121   24.749
H   49.188   32.393   24.041
H   49.247   30.580   22.363
H   47.873   31.721   21.942
H   47.553   30.191   22.702
H   45.432   28.548   26.188
H   43.766   28.333   26.390
H   44.933   27.769   27.699
H   42.744   31.003   31.150
H   43.510   29.876   35.150
H   42.294   30.992   35.297
H   43.976   31.604   35.279
H   48.223   34.996   37.921
H   46.601   35.357   37.741
H   48.581   37.618   37.173
H   44.487   38.074   38.107
H   45.310   37.973   39.698
H   44.995   36.643   38.573
H   46.193   40.102   37.889
H   47.816   39.711   37.258
H   47.809   39.216   40.173
H   46.972   40.709   39.976
H   49.509   40.359   38.409
H   49.528   40.983   40.066
H   47.478   42.352   38.511
H   48.240   42.988   36.681
H   49.458   41.810   36.584
H   49.769   43.504   37.267
H   49.591   44.099   39.343
H   49.102   42.994   40.664
H   47.090   43.899   40.808
H   46.903   44.223   39.026
H   48.905   46.053   40.212
H   47.606   45.868   41.298
H   46.928   46.365   38.345
H   47.018   48.736   38.705
H   48.365   47.918   38.162
H   48.231   48.416   39.947
H   45.495   47.823   40.552
H   45.430   46.051   40.574
H   45.028   47.411   37.830
H   44.086   47.991   38.919
H   43.811   45.182   39.521
H   44.166   45.282   37.814
H   41.942   47.125   38.255
H   42.518   46.631   40.840
H   41.298   45.402   40.388
H   40.953   47.020   40.019
H   41.442   45.166   36.890
H   40.434   44.947   38.382
H   41.971   44.103   38.094

