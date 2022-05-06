%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_501_chromophore_25 TDDFT with blyp functional

0 1
Mg   -2.331   34.278   27.067
C   -3.293   32.735   29.992
C   -0.779   36.821   29.040
C   -1.689   35.993   24.345
C   -4.005   31.861   25.191
N   -2.106   34.647   29.264
C   -2.548   33.871   30.258
C   -2.044   34.455   31.618
C   -1.357   35.846   31.238
C   -1.460   35.808   29.701
C   -2.074   37.110   31.672
C   -1.162   33.504   32.441
C   -1.659   33.386   33.872
C   -0.699   33.035   35.018
O   0.371   32.410   34.916
O   -1.083   33.614   36.234
N   -1.296   36.109   26.753
C   -0.702   36.964   27.636
C   -0.065   38.028   26.998
C   -0.403   37.907   25.605
C   -1.159   36.611   25.503
C   0.745   39.095   27.720
C   -0.121   38.872   24.465
O   -0.653   38.655   23.359
C   0.658   40.108   24.662
N   -2.762   33.931   25.110
C   -2.339   34.762   24.093
C   -2.785   34.313   22.652
C   -3.277   32.840   22.985
C   -3.378   32.872   24.521
C   -3.869   35.120   21.834
C   -2.504   31.610   22.324
C   -1.187   31.253   22.961
N   -3.528   32.729   27.430
C   -4.160   31.811   26.613
C   -4.853   30.832   27.346
C   -4.498   31.150   28.662
C   -3.680   32.317   28.688
C   -5.574   29.729   26.696
C   -4.807   30.738   29.977
O   -5.598   29.859   30.325
C   -4.009   31.740   30.964
C   -4.958   32.384   31.855
O   -5.732   33.270   31.573
O   -4.956   31.788   33.041
C   -5.912   32.195   34.026
C   -0.050   33.741   37.259
C   -0.589   34.578   38.310
C   -1.491   34.265   39.240
C   -2.169   32.904   39.471
C   -1.969   35.410   40.160
C   -1.195   35.647   41.488
C   -1.161   37.108   41.933
C   -2.102   37.433   43.127
C   -3.193   38.463   42.769
C   -1.265   37.859   44.374
C   -0.483   36.594   44.925
C   1.006   36.885   45.281
C   1.444   36.073   46.566
C   0.871   36.657   47.793
C   3.056   36.083   46.642
C   3.639   34.790   46.090
C   4.773   34.995   45.028
C   4.858   33.875   44.000
C   6.247   33.238   43.839
C   4.434   34.298   42.580
H   -0.353   37.619   29.651
H   -1.553   36.448   23.362
H   -4.518   31.127   24.566
H   -2.999   34.657   32.102
H   -0.413   35.856   31.784
H   -2.948   36.942   32.302
H   -2.458   37.642   30.802
H   -1.356   37.818   32.087
H   -0.102   33.758   32.458
H   -1.189   32.529   31.955
H   -2.346   32.539   33.878
H   -2.140   34.337   34.100
H   0.948   38.816   28.754
H   0.132   39.991   27.820
H   1.620   39.384   27.139
H   0.440   40.833   23.877
H   1.709   39.839   24.552
H   0.432   40.548   25.633
H   -1.935   34.126   21.995
H   -4.260   32.697   22.538
H   -4.171   35.974   22.439
H   -4.797   34.547   21.821
H   -3.622   35.439   20.821
H   -2.252   31.926   21.312
H   -3.115   30.717   22.198
H   -1.060   30.178   22.838
H   -0.969   31.525   23.993
H   -0.422   31.795   22.405
H   -5.209   29.357   25.739
H   -6.660   29.812   26.677
H   -5.438   28.937   27.432
H   -3.308   31.142   31.546
H   -5.544   33.011   34.649
H   -6.056   31.346   34.694
H   -6.831   32.520   33.538
H   0.829   34.240   36.850
H   0.257   32.738   37.556
H   -0.189   35.579   38.149
H   -1.750   32.098   38.869
H   -3.154   33.020   39.019
H   -2.271   32.599   40.512
H   -2.984   35.161   40.470
H   -1.948   36.352   39.612
H   -0.144   35.378   41.382
H   -1.603   35.124   42.353
H   -1.450   37.780   41.125
H   -0.095   37.227   42.128
H   -2.729   36.572   43.360
H   -3.079   39.276   43.486
H   -4.160   38.010   42.985
H   -3.293   38.846   41.753
H   -1.978   38.222   45.115
H   -0.557   38.658   44.153
H   -0.419   35.862   44.120
H   -0.920   36.148   45.819
H   1.250   37.948   45.271
H   1.633   36.619   44.429
H   1.057   35.063   46.432
H   0.064   35.997   48.113
H   0.393   37.630   47.674
H   1.511   36.922   48.635
H   3.352   36.011   47.689
H   3.526   36.977   46.233
H   2.893   34.170   45.593
H   3.998   34.209   46.940
H   5.608   34.963   45.728
H   4.915   35.982   44.588
H   4.232   33.030   44.286
H   6.708   33.471   42.879
H   6.136   32.168   44.017
H   6.903   33.619   44.622
H   3.574   34.968   42.588
H   4.122   33.518   41.885
H   5.226   34.855   42.080

