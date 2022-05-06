%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1051_chromophore_13 TDDFT with blyp functional

0 1
Mg   46.673   24.658   28.683
C   47.267   26.895   31.432
C   45.521   22.354   30.995
C   46.715   22.362   26.318
C   48.396   27.059   26.680
N   46.287   24.759   30.919
C   46.724   25.660   31.869
C   46.392   25.272   33.255
C   46.086   23.771   33.086
C   45.912   23.598   31.566
C   47.192   22.873   33.699
C   45.122   26.139   33.670
C   45.317   27.100   34.908
C   44.345   27.054   36.103
O   43.140   26.797   36.003
O   44.972   27.377   37.299
N   46.099   22.590   28.575
C   45.655   21.890   29.668
C   45.265   20.556   29.081
C   45.521   20.553   27.678
C   46.128   21.810   27.454
C   44.596   19.563   30.017
C   45.330   19.474   26.631
O   45.664   19.613   25.469
C   44.574   18.226   27.059
N   47.451   24.662   26.831
C   47.262   23.604   26.003
C   47.790   24.016   24.603
C   48.141   25.589   24.673
C   47.963   25.844   26.182
C   48.963   23.156   24.038
C   47.167   26.374   23.790
C   45.858   26.806   24.409
N   47.647   26.638   28.905
C   48.329   27.431   28.041
C   48.835   28.612   28.616
C   48.356   28.485   29.919
C   47.668   27.242   30.054
C   49.704   29.665   27.972
C   48.322   29.177   31.219
O   48.769   30.243   31.530
C   47.702   28.191   32.209
C   48.702   27.942   33.272
O   49.712   27.356   33.051
O   48.384   28.570   34.465
C   49.327   28.283   35.539
C   44.102   27.341   38.469
C   44.202   26.023   39.206
C   43.979   25.768   40.474
C   43.460   26.776   41.490
C   44.545   24.484   40.965
C   43.605   23.343   40.937
C   42.781   23.307   42.227
C   41.319   22.793   42.207
C   40.918   21.857   40.984
C   40.290   23.969   42.247
C   39.172   23.690   43.417
C   39.029   24.784   44.447
C   39.159   24.178   45.867
C   37.783   24.025   46.572
C   40.192   24.983   46.719
C   41.670   24.544   46.418
C   42.192   23.383   47.255
C   43.393   23.801   48.138
C   44.667   22.994   47.755
C   43.159   23.797   49.684
H   45.142   21.637   31.726
H   46.743   21.690   25.458
H   48.825   27.794   25.996
H   47.233   25.460   33.923
H   45.150   23.403   33.506
H   47.715   22.504   32.817
H   46.801   22.021   34.255
H   47.826   23.474   34.350
H   44.176   25.652   33.906
H   44.958   26.802   32.821
H   45.217   28.127   34.558
H   46.324   27.055   35.323
H   43.599   19.261   29.694
H   44.344   20.067   30.950
H   45.220   18.713   30.293
H   45.116   17.621   27.786
H   44.277   17.668   26.171
H   43.586   18.511   27.420
H   46.991   23.952   23.864
H   49.126   25.848   24.286
H   49.976   23.558   24.071
H   48.805   22.747   23.040
H   48.945   22.344   24.766
H   46.956   25.852   22.857
H   47.688   27.289   23.509
H   45.047   26.678   23.693
H   45.907   27.893   24.468
H   45.538   26.257   25.295
H   50.051   29.448   26.962
H   50.573   29.845   28.605
H   49.188   30.623   28.029
H   46.831   28.736   32.573
H   49.742   27.278   35.469
H   48.773   28.348   36.476
H   50.083   29.065   35.618
H   43.025   27.506   38.487
H   44.424   28.109   39.172
H   44.724   25.249   38.644
H   42.561   26.379   41.962
H   43.260   27.753   41.051
H   44.136   26.957   42.326
H   44.896   24.802   41.946
H   45.430   24.226   40.383
H   44.017   22.391   40.602
H   42.851   23.564   40.181
H   42.876   24.203   42.840
H   43.410   22.672   42.851
H   41.177   22.217   43.121
H   40.902   20.820   41.318
H   41.518   22.006   40.086
H   39.907   22.123   40.678
H   39.755   24.198   41.326
H   40.774   24.919   42.475
H   39.204   22.680   43.825
H   38.222   23.675   42.882
H   38.008   25.131   44.291
H   39.749   25.593   44.323
H   39.638   23.208   45.730
H   37.954   23.316   47.382
H   37.143   23.383   45.967
H   37.240   24.944   46.795
H   40.091   24.880   47.800
H   40.064   26.055   46.568
H   42.395   25.357   46.375
H   41.599   24.202   45.386
H   42.503   22.592   46.573
H   41.342   22.965   47.793
H   43.676   24.850   48.053
H   45.395   22.999   48.566
H   45.116   23.469   46.883
H   44.270   21.995   47.576
H   42.230   23.316   49.993
H   43.151   24.800   50.110
H   43.971   23.293   50.208
