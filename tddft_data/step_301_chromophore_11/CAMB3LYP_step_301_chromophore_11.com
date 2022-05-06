%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_301_chromophore_11 TDDFT with cam-b3lyp functional

0 1
Mg   53.356   24.257   43.946
C   50.564   26.451   43.507
C   51.230   21.510   43.067
C   56.010   22.345   43.796
C   55.245   27.034   44.826
N   51.135   24.061   42.985
C   50.244   25.116   43.092
C   48.796   24.516   42.858
C   49.013   22.951   43.059
C   50.546   22.820   43.021
C   48.268   22.319   44.301
C   48.122   24.815   41.494
C   49.019   24.850   40.166
C   48.343   24.162   38.938
O   47.945   24.762   37.958
O   48.263   22.822   39.123
N   53.607   22.254   43.514
C   52.636   21.268   43.224
C   53.213   19.972   43.248
C   54.598   20.148   43.504
C   54.806   21.604   43.562
C   52.418   18.685   42.999
C   55.702   19.027   43.619
O   55.324   17.805   43.419
C   57.121   19.247   43.906
N   55.364   24.605   44.104
C   56.325   23.655   43.954
C   57.712   24.306   44.214
C   57.453   25.869   44.317
C   55.952   25.893   44.468
C   58.417   23.729   45.470
C   57.832   26.809   43.071
C   58.596   28.023   43.444
N   53.008   26.285   44.231
C   53.888   27.271   44.617
C   53.145   28.489   44.812
C   51.840   28.175   44.493
C   51.809   26.845   44.082
C   53.845   29.666   45.162
C   50.492   28.748   44.365
O   50.177   29.870   44.628
C   49.642   27.610   43.745
C   49.078   28.153   42.450
O   49.757   28.498   41.479
O   47.725   28.065   42.560
C   46.902   28.453   41.323
C   47.679   22.185   37.951
C   48.647   22.011   36.750
C   48.543   21.096   35.804
C   47.428   20.079   35.774
C   49.664   20.831   34.828
C   50.699   19.833   35.415
C   52.222   19.972   34.819
C   52.762   18.576   34.575
C   54.255   18.413   34.941
C   52.522   17.954   33.201
C   52.078   16.518   33.107
C   53.322   15.501   33.102
C   53.214   14.336   32.055
C   52.543   13.160   32.792
C   54.633   13.905   31.456
C   54.936   14.430   30.039
C   55.436   13.350   29.020
C   54.210   12.970   28.142
C   54.083   11.507   28.027
C   53.890   13.761   26.884
H   50.573   20.642   43.152
H   56.921   21.743   43.815
H   55.887   27.913   44.912
H   48.033   24.883   43.544
H   48.644   22.388   42.201
H   47.347   21.864   43.935
H   48.069   23.123   45.010
H   48.875   21.528   44.743
H   47.557   25.747   41.491
H   47.444   23.999   41.241
H   49.987   24.414   40.412
H   49.251   25.884   39.914
H   52.868   18.172   42.149
H   51.369   18.829   42.739
H   52.403   17.883   43.737
H   57.696   19.724   43.113
H   57.528   18.261   44.128
H   57.207   19.786   44.850
H   58.393   24.103   43.387
H   57.948   26.197   45.232
H   59.433   23.364   45.321
H   57.827   22.880   45.817
H   58.303   24.350   46.358
H   56.903   27.070   42.563
H   58.379   26.242   42.318
H   59.645   27.950   43.155
H   58.591   28.260   44.508
H   58.082   28.854   42.962
H   54.399   29.404   46.063
H   53.122   30.421   45.473
H   54.654   29.922   44.478
H   48.913   27.279   44.485
H   46.894   27.637   40.601
H   47.434   29.290   40.873
H   45.859   28.749   41.437
H   46.668   22.515   37.711
H   47.449   21.205   38.368
H   49.601   22.538   36.762
H   47.529   19.126   36.293
H   47.133   19.807   34.761
H   46.490   20.554   36.063
H   50.217   21.754   34.655
H   49.366   20.359   33.892
H   50.207   18.875   35.247
H   50.710   20.043   36.484
H   52.834   20.645   35.420
H   52.146   20.398   33.819
H   52.341   17.829   35.248
H   54.347   17.830   35.857
H   54.765   19.373   35.027
H   54.791   17.780   34.234
H   53.328   18.258   32.534
H   51.655   18.414   32.727
H   51.639   16.405   32.115
H   51.320   16.234   33.837
H   53.367   15.051   34.094
H   54.255   16.056   33.004
H   52.518   14.480   31.228
H   53.391   12.635   33.233
H   52.046   12.506   32.075
H   51.821   13.476   33.545
H   54.659   12.818   31.381
H   55.433   14.289   32.089
H   55.737   15.150   30.201
H   54.106   15.052   29.702
H   55.874   12.517   29.570
H   56.119   13.938   28.406
H   53.335   13.252   28.727
H   54.859   11.067   28.654
H   54.192   11.173   26.995
H   53.058   11.295   28.330
H   54.599   14.530   26.580
H   52.929   14.275   26.909
H   53.759   13.138   25.999

