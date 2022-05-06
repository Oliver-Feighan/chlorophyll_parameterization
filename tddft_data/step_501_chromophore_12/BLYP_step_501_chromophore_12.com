%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_501_chromophore_12 TDDFT with blyp functional

0 1
Mg   46.936   15.625   28.058
C   44.932   15.296   30.937
C   49.090   17.559   29.833
C   48.760   16.004   25.235
C   44.458   14.021   26.229
N   46.958   16.417   30.082
C   46.097   16.059   31.123
C   46.628   16.601   32.416
C   47.629   17.680   31.939
C   47.921   17.182   30.500
C   47.128   19.158   31.891
C   47.277   15.546   33.389
C   46.823   15.653   34.810
C   47.918   15.572   35.890
O   48.554   14.555   36.161
O   48.138   16.740   36.624
N   48.693   16.535   27.656
C   49.516   17.231   28.557
C   50.692   17.655   27.908
C   50.642   17.200   26.507
C   49.366   16.549   26.420
C   51.746   18.547   28.496
C   51.622   17.408   25.272
O   51.379   16.913   24.197
C   52.942   18.148   25.435
N   46.590   15.180   25.997
C   47.507   15.463   25.036
C   46.956   15.017   23.598
C   45.704   14.106   23.965
C   45.552   14.449   25.508
C   46.571   16.163   22.622
C   45.923   12.573   23.705
C   44.848   12.011   22.873
N   45.055   14.860   28.457
C   44.202   14.133   27.628
C   43.059   13.685   28.376
C   43.291   14.140   29.658
C   44.479   14.800   29.683
C   41.946   12.804   27.751
C   42.745   14.209   31.001
O   41.650   13.856   31.380
C   43.810   15.006   31.906
C   43.160   16.206   32.464
O   42.630   17.116   31.774
O   43.204   16.125   33.825
C   42.271   17.048   34.524
C   49.085   16.573   37.754
C   49.246   17.810   38.592
C   50.486   18.118   39.058
C   51.768   17.333   38.861
C   50.745   19.482   39.719
C   51.698   20.567   39.125
C   51.246   21.847   39.614
C   52.263   22.999   39.112
C   53.193   23.439   40.228
C   51.621   24.225   38.391
C   51.675   24.145   36.844
C   52.768   25.003   36.156
C   53.608   24.299   35.106
C   54.987   23.750   35.693
C   53.935   25.213   33.887
C   52.808   25.092   32.722
C   53.165   24.787   31.278
C   52.285   23.646   30.652
C   50.803   23.969   30.341
C   52.379   22.365   31.513
H   49.713   18.206   30.454
H   49.276   16.094   24.277
H   43.659   13.583   25.628
H   45.898   17.172   32.991
H   48.538   17.726   32.539
H   46.412   19.408   32.675
H   46.696   19.313   30.902
H   47.905   19.922   31.896
H   48.352   15.689   33.281
H   47.120   14.526   33.037
H   46.116   14.825   34.864
H   46.374   16.632   34.977
H   52.688   18.061   28.751
H   51.399   18.922   29.458
H   52.048   19.431   27.935
H   53.468   18.018   24.490
H   53.472   17.832   26.334
H   52.751   19.177   25.741
H   47.741   14.502   23.044
H   44.840   14.473   23.409
H   46.632   17.178   23.016
H   45.554   16.075   22.239
H   47.247   16.086   21.770
H   45.878   12.050   24.661
H   46.850   12.328   23.187
H   45.207   11.939   21.846
H   43.863   12.476   22.893
H   44.583   11.022   23.247
H   42.319   12.370   26.823
H   41.000   13.343   27.682
H   41.837   11.979   28.454
H   44.196   14.384   32.714
H   42.260   16.836   35.593
H   41.279   16.758   34.177
H   42.557   18.060   34.236
H   50.116   16.369   37.464
H   48.887   15.660   38.315
H   48.436   18.538   38.635
H   52.108   17.062   37.862
H   51.615   16.352   39.310
H   52.529   17.803   39.485
H   50.865   19.184   40.761
H   49.780   19.980   39.625
H   51.867   20.589   38.049
H   52.675   20.282   39.516
H   51.026   22.016   40.668
H   50.307   22.065   39.106
H   52.955   22.591   38.375
H   54.148   23.621   39.735
H   53.455   22.659   40.943
H   52.878   24.326   40.776
H   52.018   25.215   38.614
H   50.597   24.282   38.759
H   50.699   24.576   36.620
H   51.790   23.075   36.667
H   53.424   25.407   36.928
H   52.368   25.897   35.679
H   53.098   23.416   34.720
H   55.248   23.037   34.912
H   54.668   23.308   36.637
H   55.806   24.456   35.831
H   54.917   25.015   33.458
H   54.019   26.247   34.220
H   52.475   26.129   32.739
H   52.001   24.456   33.087
H   54.200   24.511   31.077
H   52.954   25.620   30.607
H   52.741   23.468   29.678
H   50.664   24.953   30.790
H   50.096   23.216   30.689
H   50.768   24.112   29.261
H   52.621   21.526   30.861
H   51.434   22.209   32.034
H   53.214   22.356   32.213

